// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"runtime"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// eg2Layer holds one EmbeddingGemma 2 (Gemma 4 text) layer used as a
// bidirectional encoder. Everything is bias-free and, unlike Gemma 3, every
// RMSNorm scales by the checkpoint weight as-is (no unit offset). Layers are
// not uniform: full-attention layers use a wider head_dim and fewer kv heads
// than sliding ones, so each layer carries its own shape. After the
// attention and MLP sublayers (Gemma's four-norm sandwich) comes the PLE
// block — a gated projection of this layer's per-layer input slice — and
// the residual stream is then multiplied by the learned layerScalar.
type eg2Layer struct {
	nkv, dh   int  // kv heads and head_dim of this layer
	global    bool // full attention (vs sliding window)
	inNormG   []float32
	qProj     denseWeight
	kProj     denseWeight
	vProj     denseWeight
	oProj     denseWeight
	qNormG    []float32 // [dh]
	kNormG    []float32 // [dh]
	postAttnG []float32
	preFFNG   []float32
	postFFNG  []float32
	gateProj  denseWeight
	upProj    denseWeight
	downProj  denseWeight
	pleGate   denseWeight // per_layer_input_gate [P×H]
	pleProj   denseWeight // per_layer_projection [H×P]
	postPLEG  []float32   // post_per_layer_input_norm [H]
	scalar    float32     // layer_scalar
}

// eg2Prefixes are the tensor-name prefixes the text backbone is found under:
// the published multimodal checkpoint ("language_model."), a nested export,
// or a text-only file with bare names.
var eg2Prefixes = []string{"language_model.", "model.language_model.", ""}

// eg2RowBlk is the query-row block of one attention work unit.
const eg2RowBlk = 64

// loadEmbeddingGemma2 builds the EmbeddingGemma 2 text path from the mmapped
// checkpoint, reading only the text tensors: the vision and audio towers in
// the same file are never decoded. The token table stays bfloat16 when the
// checkpoint stores it so (262144×512 — 512 MB as float32).
func loadEmbeddingGemma2(weightsPath string, cfg Config, quantize QuantMode, workers int) (*Model, error) {
	r, err := safetensors.OpenReader(weightsPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = r.Close() }()

	pre := ""
	found := false
	for _, p := range eg2Prefixes {
		if _, ok := r.Shape(p + "embed_tokens.weight"); ok {
			pre, found = p, true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("weights %s: no embedding_gemma2 text tensors (embed_tokens.weight under %q)", weightsPath, eg2Prefixes)
	}
	f32 := func(name string, shape ...int) ([]float32, error) {
		v, err := r.F32(pre+name, shape...)
		if err != nil {
			return nil, fmt.Errorf("weights %s: %w", weightsPath, err)
		}
		return v, nil
	}

	H, I, P := cfg.HiddenSize, cfg.IntermediateSize, cfg.PerLayerInputDim
	nq, L := cfg.NumAttentionHeads, cfg.NumHiddenLayers
	m := &Model{cfg: cfg, workers: workers, posOff: 0}
	m.scratchPool.New = func() any { return new(scratch) }

	emb, isBF16, err := r.BF16(pre+"embed_tokens.weight", cfg.VocabSize, H)
	switch {
	case err != nil:
		return nil, fmt.Errorf("weights %s: %w", weightsPath, err)
	case isBF16:
		m.wordEmbBF16 = emb
	default:
		if m.wordEmb, err = f32("embed_tokens.weight", cfg.VocabSize, H); err != nil {
			return nil, err
		}
	}
	pleRaw, err := f32("ple.per_layer_model_projection.weight", L*P, H)
	if err != nil {
		return nil, err
	}
	m.eg2PLEProj = newDense(pleRaw, nil, H, L*P, quantize)
	if m.eg2PLENorm, err = f32("ple.per_layer_projection_norm.weight", P); err != nil {
		return nil, err
	}
	if m.finalNormG, err = f32("norm.weight", H); err != nil {
		return nil, err
	}
	if m.eg2EmbProj, err = f32("embedding_projection.weight", cfg.EmbeddingDim, H); err != nil {
		return nil, err
	}

	dhMax := 0
	m.eg2Layers = make([]eg2Layer, L)
	for i := range m.eg2Layers {
		l := &m.eg2Layers[i]
		l.nkv, l.dh, l.global = cfg.LayerKVHeads[i], cfg.LayerHeadDim[i], cfg.LayerGlobal[i]
		dhMax = max(dhMax, l.dh)
		qDim, kvDim := nq*l.dh, l.nkv*l.dh
		p := fmt.Sprintf("layers.%d.", i)
		for _, nm := range []struct {
			dst  *[]float32
			name string
			dim  int
		}{
			{&l.inNormG, "input_layernorm.weight", H},
			{&l.qNormG, "self_attn.q_norm.weight", l.dh},
			{&l.kNormG, "self_attn.k_norm.weight", l.dh},
			{&l.postAttnG, "post_attention_layernorm.weight", H},
			{&l.preFFNG, "pre_feedforward_layernorm.weight", H},
			{&l.postFFNG, "post_feedforward_layernorm.weight", H},
			{&l.postPLEG, "ple_block.post_per_layer_input_norm.weight", H},
		} {
			if *nm.dst, err = f32(p+nm.name, nm.dim); err != nil {
				return nil, err
			}
		}
		for _, pr := range []struct {
			dst     *denseWeight
			name    string
			in, out int
		}{
			{&l.qProj, "self_attn.q_proj.weight", H, qDim},
			{&l.kProj, "self_attn.k_proj.weight", H, kvDim},
			{&l.vProj, "self_attn.v_proj.weight", H, kvDim},
			{&l.oProj, "self_attn.o_proj.weight", qDim, H},
			{&l.gateProj, "mlp.gate_proj.weight", H, I},
			{&l.upProj, "mlp.up_proj.weight", H, I},
			{&l.downProj, "mlp.down_proj.weight", I, H},
			{&l.pleGate, "ple_block.per_layer_input_gate.weight", H, P},
			{&l.pleProj, "ple_block.per_layer_projection.weight", P, H},
		} {
			raw, err := f32(p+pr.name, pr.out, pr.in)
			if err != nil {
				return nil, err
			}
			*pr.dst = newDense(raw, nil, pr.in, pr.out, quantize)
		}
		sc, err := f32(p+"layer_scalar", 1)
		if err != nil {
			return nil, err
		}
		l.scalar = sc[0]
	}
	// The value RMSNorm carries no weight (with_scale=False): a shared
	// all-ones gamma lets the standard kernel run unchanged.
	m.eg2Ones = make([]float32, dhMax)
	for i := range m.eg2Ones {
		m.eg2Ones[i] = 1
	}
	return m, nil
}

// embedRow writes token id's embedding row, scaled, into dst.
func (m *Model) eg2EmbedRow(dst []float32, id int, scale float32) {
	H := m.cfg.HiddenSize
	if m.wordEmbBF16 != nil {
		src := m.wordEmbBF16[id*H : id*H+H]
		for j := range dst {
			dst[j] = math.Float32frombits(uint32(src[j])<<16) * scale
		}
		return
	}
	src := m.wordEmb[id*H : id*H+H]
	for j := range dst {
		dst[j] = src[j] * scale
	}
}

// eg2Project applies embedding_projection (bias-free, [E×H]) to one H-vector.
// The reference projects every token before mean pooling; the projection is
// linear, so projecting the pooled vector once is the same result.
func (m *Model) eg2Project(v []float32) []float32 {
	H, E := m.cfg.HiddenSize, m.cfg.EmbeddingDim
	out := make([]float32, E)
	for o := range E {
		row := m.eg2EmbProj[o*H : o*H+H]
		var acc float32
		for i, w := range row {
			acc += w * v[i]
		}
		out[o] = acc
	}
	return out
}

// encodeEmbeddingGemma2 runs the Gemma 4 text backbone as a bidirectional
// encoder: √H-scaled token embeddings; the per-layer inputs (one projection
// of the embeddings, scaled by 1/√H and RMS-normed per layer slice); then
// for every layer the attention sublayer (QK-norm, weightless V-norm, RoPE
// with the layer type's theta, scale 1.0, GQA, symmetric |i−j| ≤ window on
// sliding layers), the GeGLU MLP, the PLE block, and the layer scalar;
// finally the trailing RMSNorm. Pooling and the 512→768 projection run in
// ForwardWorkers.
func (m *Model) encodeEmbeddingGemma2(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	cfg := &m.cfg
	H, I, P := cfg.HiddenSize, cfg.IntermediateSize, cfg.PerLayerInputDim
	nq, L := cfg.NumAttentionHeads, len(m.eg2Layers)
	eps := cfg.LayerNormEps
	window := cfg.SlidingWindow

	dhG, dhL, qDimMax, kvDimMax := 0, 0, 0, 0
	for i := range m.eg2Layers {
		l := &m.eg2Layers[i]
		if l.global {
			dhG = l.dh
		} else {
			dhL = l.dh
		}
		qDimMax = max(qDimMax, nq*l.dh)
		kvDimMax = max(kvDimMax, l.nkv*l.dh)
	}

	s := m.scratchPool.Get().(*scratch)
	committed := false
	defer func() {
		if !committed {
			m.scratchPool.Put(s)
		}
	}()

	mPad := tensor.PackAPad(seq)
	inMax := max(qDimMax, I, H, P)
	s.x = grow(s.x, seq*H)
	s.normed = grow(s.normed, seq*H)
	s.qwQ = grow(s.qwQ, mPad*qDimMax)
	s.qwK = grow(s.qwK, mPad*kvDimMax)
	s.qwV = grow(s.qwV, mPad*kvDimMax)
	s.qwKHead = grow(s.qwKHead, kvDimMax*seq) // per-kv-head K (row-major), normed+RoPE'd
	s.qwVHead = grow(s.qwVHead, kvDimMax*seq) // per-kv-head V TRANSPOSED [dh×seq] for P·V
	s.qwQHead = grow(s.qwQHead, qDimMax*seq)
	s.qwCHead = grow(s.qwCHead, qDimMax*seq)
	s.qwCtx = grow(s.qwCtx, seq*qDimMax)
	s.scores = grow(s.scores, nq*seq*seq)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.wiOut = grow(s.wiOut, mPad*I) // GeGLU gate projection
	s.qwUp = grow(s.qwUp, mPad*I)   // GeGLU up projection
	s.geglu = grow(s.geglu, seq*I)  // gelu_tanh(gate)*up
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.eg2PLE = grow(s.eg2PLE, mPad*L*P)
	s.eg2Gate = grow(s.eg2Gate, mPad*P)
	s.aPack = grow(s.aPack, mPad*inMax)
	s.cosG, s.sinG = grow(s.cosG, seq*dhG/2), grow(s.sinG, seq*dhG/2)
	s.cosL, s.sinL = grow(s.cosL, seq*dhL/2), grow(s.sinL, seq*dhL/2)
	kgMax := (inMax + 3) / 4
	if cap(s.qact) < mPad*kgMax*4 {
		s.qact = make([]uint8, mPad*kgMax*4)
	}
	s.qact = s.qact[:mPad*kgMax*4]
	s.ascales = grow(s.ascales, mPad)

	s.fanout = runtime.GOMAXPROCS(0)
	if workers > 0 {
		s.fanout = min(s.fanout, workers)
	}
	s.pool = tensor.NewPool(s.fanout - 1)
	defer s.pool.Stop()

	if dhG > 0 {
		ropeTable(s.cosG, s.sinG, seq, dhG, cfg.GlobalRopeTheta)
	}
	if dhL > 0 {
		ropeTable(s.cosL, s.sinL, seq, dhL, cfg.LocalRopeTheta)
	}

	embScale := float32(math.Sqrt(float64(H)))
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, cfg.VocabSize)
		}
		m.eg2EmbedRow(x[i*H:i*H+H], int(id), embScale)
	}

	// Per-layer inputs: one projection of the scaled embeddings to L slices
	// of P, scaled by 1/√H, each slice RMS-normed with one shared weight.
	// Rows are [seq][L][P], so the norm runs over seq·L contiguous rows.
	ple := s.eg2PLE
	m.applyDense(ple, x[:seq*H], &m.eg2PLEProj, seq, s)
	pleScale := float32(1 / math.Sqrt(float64(H)))
	for j := range ple[:seq*L*P] {
		ple[j] *= pleScale
	}
	tensor.RMSNorm(ple[:seq*L*P], m.eg2PLENorm, seq*L, P, eps)

	normed := s.normed
	q, k, v := s.qwQ, s.qwK, s.qwV
	kHead, vHeadT := s.qwKHead, s.qwVHead
	qHead, cHead := s.qwQHead, s.qwCHead
	ctx := s.qwCtx
	scores := s.scores
	attnOut := s.attnOut
	gate, up, act, down := s.wiOut, s.qwUp, s.geglu, s.ffnOut
	pg := s.eg2Gate

	for li := range m.eg2Layers {
		l := &m.eg2Layers[li]
		dh, nkv := l.dh, l.nkv
		qDim, kvDim := nq*dh, nkv*dh
		group := nq / nkv
		cos, sin := s.cosL, s.sinL
		if l.global {
			cos, sin = s.cosG, s.sinG
		}

		// --- Attention block (pre-norm, bidirectional, GQA, QK/V-norm) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.inNormG, seq, H, eps)
		m.applyDense(q, normed[:seq*H], &l.qProj, seq, s)
		m.applyDense(k, normed[:seq*H], &l.kProj, seq, s)
		m.applyDense(v, normed[:seq*H], &l.vProj, seq, s)

		// Per kv-head (shared across a GQA group): repack + norm + RoPE K;
		// weightless RMS-norm V and transpose it to [dh×seq] for score·V.
		s.pool.Run(nkv, func(kv int) {
			kH := kHead[kv*seq*dh : (kv+1)*seq*dh]
			vHT := vHeadT[kv*dh*seq : (kv+1)*dh*seq]
			for i := range seq {
				copy(kH[i*dh:i*dh+dh], k[i*kvDim+kv*dh:i*kvDim+kv*dh+dh])
				vRow := v[i*kvDim+kv*dh : i*kvDim+kv*dh+dh]
				tensor.RMSNorm(vRow, m.eg2Ones, 1, dh, eps)
				for d := range dh {
					vHT[d*seq+i] = vRow[d]
				}
			}
			tensor.RMSNorm(kH, l.kNormG, seq, dh, eps)
			applyRoPE(kH, seq, dh, cos, sin)
		})

		// Per query-head: repack + norm + RoPE Q.
		s.pool.Run(nq, func(hh int) {
			qH := qHead[hh*seq*dh : (hh+1)*seq*dh]
			for i := range seq {
				copy(qH[i*dh:i*dh+dh], q[i*qDim+hh*dh:i*qDim+hh*dh+dh])
			}
			tensor.RMSNorm(qH, l.qNormG, seq, dh, eps)
			applyRoPE(qH, seq, dh, cos, sin)
		})

		// Unscaled bidirectional attention (QK-norm makes 1/√dh unnecessary
		// — scaling = 1.0) of each query head against its kv head,
		// sliding-window masked on local layers. Query rows are independent
		// (softmax is per row), so the work fans out over (head, row block)
		// units: with only 4 heads, per-head units would leave most cores
		// idle on long inputs.
		nb := (seq + eg2RowBlk - 1) / eg2RowBlk
		s.pool.Run(nq*nb, func(u int) {
			hh, r0 := u/nb, (u%nb)*eg2RowBlk
			r1 := min(seq, r0+eg2RowBlk)
			rows := r1 - r0
			kv := hh / group
			qH := qHead[hh*seq*dh+r0*dh : hh*seq*dh+r1*dh]
			kH := kHead[kv*seq*dh : (kv+1)*seq*dh]
			vHT := vHeadT[kv*dh*seq : (kv+1)*dh*seq]
			chH := cHead[hh*seq*dh+r0*dh : hh*seq*dh+r1*dh]
			sc := scores[hh*seq*seq+r0*seq : hh*seq*seq+r1*seq]
			tensor.MatMulSerial(sc, qH, kH, rows, dh, seq)
			if !l.global {
				// Symmetric, inclusive window: a query attends where
				// |i−j| ≤ window. Global layers see every position.
				for r := range rows {
					i := r0 + r
					scRow := sc[r*seq : r*seq+seq]
					for j := range scRow {
						if j < i-window || j > i+window {
							scRow[j] = -1e30
						}
					}
				}
			}
			tensor.Softmax(sc, rows, seq)
			tensor.MatMulSerial(chH, sc, vHT, rows, seq, dh)
			for r := range rows {
				i := r0 + r
				copy(ctx[i*qDim+hh*dh:i*qDim+hh*dh+dh], chH[r*dh:r*dh+dh])
			}
		})

		m.applyDense(attnOut, ctx[:seq*qDim], &l.oProj, seq, s)
		tensor.RMSNorm(attnOut, l.postAttnG, seq, H, eps)
		tensor.Add(x, attnOut[:seq*H])

		// --- MLP block (pre-norm GeGLU, post-normed output) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.preFFNG, seq, H, eps)
		m.applyDense(gate, normed[:seq*H], &l.gateProj, seq, s)
		m.applyDense(up, normed[:seq*H], &l.upProj, seq, s)
		s.pool.Run(seq, func(i int) {
			g := gate[i*I : i*I+I]
			u := up[i*I : i*I+I]
			d := act[i*I : i*I+I]
			copy(d, g)
			tensor.GELUTanh(d)
			for j := range d {
				d[j] *= u[j]
			}
		})
		m.applyDense(down, act[:seq*I], &l.downProj, seq, s)
		tensor.RMSNorm(down, l.postFFNG, seq, H, eps)
		tensor.Add(x, down[:seq*H])

		// --- PLE block: gelu_tanh(gate(h)) ⊙ this layer's input slice,
		// projected back to H, post-normed, added; then the layer scalar.
		m.applyDense(pg, x[:seq*H], &l.pleGate, seq, s)
		s.pool.Run(seq, func(i int) {
			g := pg[i*P : i*P+P]
			tensor.GELUTanh(g)
			in := ple[(i*L+li)*P : (i*L+li)*P+P]
			for j := range g {
				g[j] *= in[j]
			}
		})
		m.applyDense(down, pg[:seq*P], &l.pleProj, seq, s)
		tensor.RMSNorm(down, l.postPLEG, seq, H, eps)
		tensor.Add(x, down[:seq*H])
		for j := range x[:seq*H] {
			x[j] *= l.scalar
		}
	}

	tensor.RMSNorm(x, m.finalNormG, seq, H, eps)

	committed = true
	return s, nil
}

// eg2Weights lists every dense weight of the EmbeddingGemma 2 path (for the
// Quantized / QuantizedActivations reports).
func (m *Model) eg2Weights() []*denseWeight {
	ws := []*denseWeight{&m.eg2PLEProj}
	for i := range m.eg2Layers {
		l := &m.eg2Layers[i]
		ws = append(ws, &l.qProj, &l.kProj, &l.vProj, &l.oProj, &l.gateProj, &l.upProj, &l.downProj, &l.pleGate, &l.pleProj)
	}
	return ws
}
