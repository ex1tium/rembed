// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
)

// eg2TextCfg is a tiny embedding_gemma2 text_config: two layers, the first
// sliding (head_dim 8, 2 kv heads), the second full attention with a wider
// head (16) and one kv head — the same heterogeneity the real checkpoint has.
func eg2TextCfg() map[string]any {
	return map[string]any{
		"model_type":                  "embedding_gemma2_text",
		"hidden_size":                 16,
		"intermediate_size":           32,
		"num_hidden_layers":           2,
		"num_attention_heads":         2,
		"num_key_value_heads":         2,
		"head_dim":                    8,
		"vocab_size":                  32,
		"max_position_embeddings":     262144,
		"rms_norm_eps":                1e-6,
		"hidden_activation":           "gelu_pytorch_tanh",
		"attention_bias":              false,
		"sliding_window":              2,
		"layer_types":                 []string{"sliding_attention", "full_attention"},
		"hidden_size_per_layer_input": 8,
		"embedding_dim":               12,
		"per_layer_config":            map[string]any{"01": map[string]any{"head_dim": 16, "num_key_value_heads": 1}},
		"rope_parameters": map[string]any{
			"full_attention":    map[string]any{"rope_theta": 1e6, "rope_type": "default"},
			"sliding_attention": map[string]any{"rope_theta": 1e4, "rope_type": "default"},
		},
	}
}

// eg2STPooling and eg2STModules are the sentence-transformers 6 spellings
// the published checkpoint ships.
const (
	eg2STPooling = `{"embedding_dimension": 12, "pooling_mode": "mean", "include_prompt": true}`
	eg2STModules = `[
 {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
 {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
 {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.base.modules.normalize.Normalize"}]`
)

func writeEG2Dir(t *testing.T, text map[string]any, pooling, modules string) string {
	t.Helper()
	dir := t.TempDir()
	cfg, err := json.Marshal(map[string]any{
		"architectures": []string{"EmbeddingGemma2Model"},
		"model_type":    "embedding_gemma2",
		"text_config":   text,
		"audio_config":  map[string]any{"model_type": "gemma4_audio", "hidden_size": 1024},
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"config.json":           string(cfg),
		"tokenizer_config.json": `{"tokenizer_class": "GemmaTokenizer"}`,
		"1_Pooling/config.json": pooling,
		"modules.json":          modules,
	} {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestDeriveEmbeddingGemma2(t *testing.T) {
	dir := writeEG2Dir(t, eg2TextCfg(), eg2STPooling, eg2STModules)
	c, err := DeriveConfig(dir, "eg2")
	if err != nil {
		t.Fatal(err)
	}
	if c.ModelType != "embedding_gemma2" || c.Tokenizer != "gemma" || c.ClsToken != "<bos>" || c.SepToken != "<eos>" {
		t.Fatalf("identity fields wrong: %+v", c)
	}
	if c.Pooling != "mean" || !c.Normalize {
		t.Fatalf("pooling=%q normalize=%v, want mean + Normalize (ST6 module names)", c.Pooling, c.Normalize)
	}
	if !slices.Equal(c.LayerGlobal, []bool{false, true}) || !slices.Equal(c.LayerHeadDim, []int{8, 16}) || !slices.Equal(c.LayerKVHeads, []int{2, 1}) {
		t.Fatalf("per-layer schedule wrong: global=%v dh=%v kv=%v", c.LayerGlobal, c.LayerHeadDim, c.LayerKVHeads)
	}
	if c.LocalRopeTheta != 1e4 || c.GlobalRopeTheta != 1e6 || c.SlidingWindow != 2 {
		t.Fatalf("rope/window wrong: local=%g global=%g window=%d", c.LocalRopeTheta, c.GlobalRopeTheta, c.SlidingWindow)
	}
	if c.PerLayerInputDim != 8 || c.EmbeddingDim != 12 || c.OutputDim() != 12 {
		t.Fatalf("PLE/embedding dims wrong: P=%d E=%d out=%d", c.PerLayerInputDim, c.EmbeddingDim, c.OutputDim())
	}
	if c.MaxSeqLen() != eg2MaxSeqLen {
		t.Fatalf("MaxSeqLen=%d, want the %d-token trained context, not the RoPE horizon", c.MaxSeqLen(), eg2MaxSeqLen)
	}
	if c.LayerNormEps != 1e-6 {
		t.Fatalf("eps=%g, want rms_norm_eps 1e-6", c.LayerNormEps)
	}
}

func TestDeriveEmbeddingGemma2Refusals(t *testing.T) {
	cases := []struct {
		name            string
		mutate          func(map[string]any)
		pooling, module string
		want            string
	}{
		{"activation", func(c map[string]any) { c["hidden_activation"] = "silu" }, "", "", "only gelu_pytorch_tanh"},
		{"attention bias", func(c map[string]any) { c["attention_bias"] = true }, "", "", "attention_bias"},
		{"softcap", func(c map[string]any) { c["attn_logit_softcapping"] = 50.0 }, "", "", "softcapping"},
		{"layer_types length", func(c map[string]any) { c["layer_types"] = []string{"full_attention"} }, "", "", "layer_types has 1 entries"},
		{"layer type", func(c map[string]any) { c["layer_types"] = []string{"chunked_attention", "full_attention"} }, "", "", "not supported"},
		{"rope type", func(c map[string]any) {
			c["rope_parameters"] = map[string]any{
				"full_attention":    map[string]any{"rope_theta": 1e6, "rope_type": "yarn"},
				"sliding_attention": map[string]any{"rope_theta": 1e4},
			}
		}, "", "", "only default RoPE"},
		{"missing rope kind", func(c map[string]any) {
			c["rope_parameters"] = map[string]any{"full_attention": map[string]any{"rope_theta": 1e6}}
		}, "", "", "lacks \"sliding_attention\""},
		{"per_layer key", func(c map[string]any) { c["per_layer_config"] = map[string]any{"x": map[string]any{"head_dim": 16}} }, "", "", "not a layer index"},
		{"odd head_dim", func(c map[string]any) { c["head_dim"] = 7 }, "", "", "even head_dim"},
		{"kv heads", func(c map[string]any) {
			c["per_layer_config"] = map[string]any{"01": map[string]any{"head_dim": 16, "num_key_value_heads": 3}}
		}, "", "", "not divisible"},
		{"mixed head_dim per type", func(c map[string]any) {
			c["num_hidden_layers"] = 3
			c["layer_types"] = []string{"sliding_attention", "sliding_attention", "full_attention"}
			c["per_layer_config"] = map[string]any{"01": map[string]any{"head_dim": 4}}
		}, "", "", "same attention type"},
		{"no embedding_dim", func(c map[string]any) { delete(c, "embedding_dim") }, "", "", "embedding_dim"},
		{"include_prompt false", nil, `{"pooling_mode": "mean", "include_prompt": false}`, "", "include_prompt=false"},
		{"pooling mode", nil, `{"pooling_mode": "max"}`, "", "unsupported pooling mode"},
		{"cls pooling", nil, `{"pooling_mode": "cls"}`, "", "requires mean pooling"},
		{"dense module", nil, "", `[{"type": "sentence_transformers.models.Dense"}]`, "unsupported sentence-transformers module"},
		{"custom module", nil, "", `[{"type": "custom_st.Transformer"}, {"type": "sentence_transformers.models.Pooling"}]`, "not a sentence_transformers module"},
		{"include_prompt string false", nil, `{"pooling_mode": "mean", "include_prompt": "false"}`, "", "include_prompt=false"},
		{"pooling spellings disagree", nil, `{"pooling_mode": "mean", "pooling_mode_cls_token": true}`, "", "contradicts pooling_mode_cls_token"},
		{"duplicate per_layer key", func(c map[string]any) {
			c["per_layer_config"] = map[string]any{"1": map[string]any{"head_dim": 16, "num_key_value_heads": 1}, "01": map[string]any{"head_dim": 16, "num_key_value_heads": 1}}
		}, "", "", "names layer 1 twice"},
		{"partial rotary", func(c map[string]any) { c["partial_rotary_factor"] = 0.25 }, "", "", "partial_rotary_factor"},
		{"kv sharing", func(c map[string]any) { c["num_kv_shared_layers"] = 4 }, "", "", "num_kv_shared_layers"},
		{"causal", func(c map[string]any) { c["use_bidirectional_attention"] = false }, "", "", "use_bidirectional_attention=false"},
		{"pooling dim mismatch", nil, `{"embedding_dimension": 64, "pooling_mode": "mean"}`, "", "embedding_dimension 64 disagrees"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			text := eg2TextCfg()
			if tc.mutate != nil {
				tc.mutate(text)
			}
			pooling, modules := eg2STPooling, eg2STModules
			if tc.pooling != "" {
				pooling = tc.pooling
			}
			if tc.module != "" {
				modules = tc.module
			}
			_, err := DeriveConfig(writeEG2Dir(t, text, pooling, modules), "eg2")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want it to mention %q", err, tc.want)
			}
		})
	}
}

// stTensor is one tensor for writeSafetensors: F32 and BF16 come from vals,
// I64 (a decoy non-float buffer) is written as zeros.
type stTensor struct {
	dtype string
	shape []int
	vals  []float32
}

func writeSafetensors(t *testing.T, path string, ts map[string]stTensor) {
	t.Helper()
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var data []byte
	for _, n := range names {
		tt := ts[n]
		elems := 1
		for _, d := range tt.shape {
			elems *= d
		}
		start := len(data)
		switch tt.dtype {
		case "F32":
			for _, v := range tt.vals {
				data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
			}
		case "BF16":
			for _, v := range tt.vals {
				data = binary.LittleEndian.AppendUint16(data, uint16(math.Float32bits(v)>>16))
			}
		case "I64":
			data = append(data, make([]byte, elems*8)...)
		}
		header[n] = map[string]any{"dtype": tt.dtype, "shape": tt.shape, "data_offsets": []int{start, len(data)}}
	}
	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hj)))
	out = append(out, hj...)
	out = append(out, data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// synthEG2 derives the tiny config (optionally mutated) and writes random
// weights for it under the multimodal "language_model." prefix, next to
// decoy vision tensors — one with a dtype the loader cannot decode.
func synthEG2(t *testing.T, mutate func(map[string]any)) (string, Config) {
	t.Helper()
	text := eg2TextCfg()
	if mutate != nil {
		mutate(text)
	}
	dir := writeEG2Dir(t, text, eg2STPooling, eg2STModules)
	cfg, err := DeriveConfig(dir, "eg2")
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	randT := func(dtype string, scale float32, shape ...int) stTensor {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
		}
		return stTensor{dtype, shape, v}
	}
	normT := func(dim int) stTensor {
		v := make([]float32, dim)
		for i := range v {
			v[i] = 1 + float32(rng.NormFloat64())*0.1
		}
		return stTensor{"F32", []int{dim}, v}
	}
	H, I, P, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.PerLayerInputDim, cfg.EmbeddingDim
	nq, L := cfg.NumAttentionHeads, cfg.NumHiddenLayers
	pre := "language_model."
	ts := map[string]stTensor{
		pre + "embed_tokens.weight":                     randT("BF16", 0.5, cfg.VocabSize, H),
		pre + "ple.per_layer_model_projection.weight":   randT("F32", 0.3, L*P, H),
		pre + "ple.per_layer_projection_norm.weight":    normT(P),
		pre + "norm.weight":                             normT(H),
		pre + "embedding_projection.weight":             randT("F32", 0.3, E, H),
		"vision_tower.patch_embedder.input_proj.weight": randT("F32", 1, 4, 4),
		"vision_tower.position_ids":                     {dtype: "I64", shape: []int{4}},
		"audio_tower.subsample.counts":                  {dtype: "I64", shape: []int{2}},
	}
	for i := range L {
		dh, kv := cfg.LayerHeadDim[i], cfg.LayerKVHeads[i]
		p := pre + "layers." + string(rune('0'+i)) + "."
		for name, dim := range map[string]int{
			"input_layernorm.weight": H, "post_attention_layernorm.weight": H,
			"pre_feedforward_layernorm.weight": H, "post_feedforward_layernorm.weight": H,
			"self_attn.q_norm.weight": dh, "self_attn.k_norm.weight": dh,
			"ple_block.post_per_layer_input_norm.weight": H,
		} {
			ts[p+name] = normT(dim)
		}
		for name, shape := range map[string][2]int{
			"self_attn.q_proj.weight": {nq * dh, H}, "self_attn.k_proj.weight": {kv * dh, H},
			"self_attn.v_proj.weight": {kv * dh, H}, "self_attn.o_proj.weight": {H, nq * dh},
			"mlp.gate_proj.weight": {I, H}, "mlp.up_proj.weight": {I, H}, "mlp.down_proj.weight": {H, I},
			"ple_block.per_layer_input_gate.weight": {P, H}, "ple_block.per_layer_projection.weight": {H, P},
		} {
			ts[p+name] = randT("F32", 0.3, shape[0], shape[1])
		}
		ts[p+"layer_scalar"] = stTensor{"F32", []int{1}, []float32{0.6}}
	}
	path := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, path, ts)
	return path, cfg
}

// TestEmbeddingGemma2SyntheticForward runs the whole load + forward path on
// random weights: the towers (including an undecodable I64 buffer) are
// skipped, the BF16 token table is gathered without widening, and the
// pooled vector is the L2-normalized EmbeddingDim projection.
func TestEmbeddingGemma2SyntheticForward(t *testing.T) {
	path, cfg := synthEG2(t, nil)
	m, err := Load(path, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	if m.wordEmbBF16 == nil || m.wordEmb != nil {
		t.Fatal("BF16 token table was widened to float32")
	}
	ids := []int64{2, 7, 19, 3, 11, 1}
	v, err := m.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != cfg.EmbeddingDim {
		t.Fatalf("pooled dim %d, want EmbeddingDim %d", len(v), cfg.EmbeddingDim)
	}
	var norm float64
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("non-finite output %v", v)
		}
		norm += float64(x) * float64(x)
	}
	if math.Abs(norm-1) > 1e-5 {
		t.Fatalf("|v|²=%g, want 1 (Normalize module)", norm)
	}
	rows, err := m.ForwardTokens(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(ids) || len(rows[0]) != cfg.EmbeddingDim {
		t.Fatalf("ForwardTokens shape %dx%d, want %dx%d (projected per token)", len(rows), len(rows[0]), len(ids), cfg.EmbeddingDim)
	}
	if _, err := m.Forward([]int64{2, 99}); err == nil {
		t.Fatal("out-of-vocab id accepted")
	}
}

// TestEmbeddingGemma2SlidingWindowLocality pins the symmetric inclusive
// window on a single-layer model: with window w, position p sees exactly
// tokens [p−w, p+w], so changing token 0 must leave every p > w bit-identical
// and must change p = w. A global layer must propagate the change everywhere.
func TestEmbeddingGemma2SlidingWindowLocality(t *testing.T) {
	oneLayer := func(kind string) func(map[string]any) {
		return func(c map[string]any) {
			c["num_hidden_layers"] = 1
			c["layer_types"] = []string{kind}
			c["per_layer_config"] = map[string]any{}
		}
	}
	ids := []int64{2, 5, 6, 7, 8, 9, 10, 1}
	alt := slices.Clone(ids)
	alt[0] = 30
	run := func(m *Model, in []int64) [][]float32 {
		rows, err := m.ForwardTokens(in)
		if err != nil {
			t.Fatal(err)
		}
		return rows
	}

	path, cfg := synthEG2(t, oneLayer("sliding_attention"))
	m, err := Load(path, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	w := cfg.SlidingWindow
	a, b := run(m, ids), run(m, alt)
	for p := range ids {
		same := slices.Equal(a[p], b[p])
		switch {
		case p > w && !same:
			t.Errorf("sliding: position %d (> window %d) changed when token 0 changed", p, w)
		case p <= w && same:
			t.Errorf("sliding: position %d (≤ window %d) did not change when token 0 changed", p, w)
		}
	}

	path, cfg = synthEG2(t, oneLayer("full_attention"))
	m, err = Load(path, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	a, b = run(m, ids), run(m, alt)
	for p := range ids {
		if slices.Equal(a[p], b[p]) {
			t.Errorf("full attention: position %d unaffected by token 0", p)
		}
	}
}

// TestEmbeddingGemma2TextOnlyNames loads a checkpoint whose text tensors
// carry no prefix (a text-only re-export).
func TestEmbeddingGemma2TextOnlyNames(t *testing.T) {
	path, cfg := synthEG2(t, nil)
	r, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(r)
	var header map[string]json.RawMessage
	if err := json.Unmarshal(r[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	stripped := map[string]json.RawMessage{}
	for k, v := range header {
		stripped[strings.TrimPrefix(k, "language_model.")] = v
	}
	hj, err := json.Marshal(stripped)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(hj)))
	out = append(out, hj...)
	out = append(out, r[8+n:]...)
	bare := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(bare, out, 0o644); err != nil {
		t.Fatal(err)
	}
	want, err := Load(path, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Load(bare, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int64{2, 4, 8, 1}
	a, _ := want.Forward(ids)
	b, _ := got.Forward(ids)
	if !slices.Equal(a, b) {
		t.Fatal("bare-name checkpoint embeds differently from the prefixed one")
	}

	if _, err := Load(filepath.Join(t.TempDir(), "missing.safetensors"), cfg, QuantNone, 1); err == nil {
		t.Fatal("missing weights file accepted")
	}
}

// TestEmbeddingGemma2ConcurrentWorkers drives the parallel paths — the
// (head, row-block) attention fan-out, the per-row PLE/GeGLU fan-outs, and
// the shared scratch pool — from several goroutines with multi-worker
// fan-outs, and requires bit-identical results to the serial forward. Run
// under -race this is the concurrency check the serial golden cannot be.
func TestEmbeddingGemma2ConcurrentWorkers(t *testing.T) {
	path, cfg := synthEG2(t, nil)
	serial, err := Load(path, cfg, QuantNone, 1)
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := Load(path, cfg, QuantNone, 4)
	if err != nil {
		t.Fatal(err)
	}
	// Lengths straddle the 64-row attention block so several row blocks
	// run per head.
	inputs := [][]int64{}
	for _, n := range []int{1, 3, 63, 64, 65, 150} {
		ids := make([]int64, n)
		for i := range ids {
			ids[i] = int64(2 + i%29)
		}
		inputs = append(inputs, ids)
	}
	want := make([][]float32, len(inputs))
	for i, ids := range inputs {
		if want[i], err = serial.Forward(ids); err != nil {
			t.Fatal(err)
		}
	}
	errs := make(chan error, 4*len(inputs))
	done := make(chan struct{})
	for g := 0; g < 4; g++ {
		go func() {
			defer func() { done <- struct{}{} }()
			for i, ids := range inputs {
				got, err := parallel.Forward(ids)
				if err != nil {
					errs <- err
					return
				}
				for j := range got {
					if math.Abs(float64(got[j]-want[i][j])) > 1e-5 {
						errs <- fmt.Errorf("input %d (len %d): dim %d = %g, serial %g", i, len(ids), j, got[j], want[i][j])
						return
					}
				}
			}
		}()
	}
	for g := 0; g < 4; g++ {
		<-done
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// TestEmbeddingGemma2ConfigDefaults: a missing rms_norm_eps is Gemma's 1e-6
// (not BERT's 1e-12); the default Gemma 4 knobs are accepted; a head count
// that does not divide hidden_size is fine (head_dim is explicit); and a
// manifest that copied the 262144 RoPE horizon is still capped at 8192.
func TestEmbeddingGemma2ConfigDefaults(t *testing.T) {
	text := eg2TextCfg()
	delete(text, "rms_norm_eps")
	text["partial_rotary_factor"] = 1.0
	text["num_kv_shared_layers"] = 0
	text["use_bidirectional_attention"] = true
	text["num_attention_heads"] = 6
	text["num_key_value_heads"] = 3
	text["per_layer_config"] = map[string]any{"1": map[string]any{"head_dim": 16, "num_key_value_heads": 1}}
	c, err := DeriveConfig(writeEG2Dir(t, text, eg2STPooling, eg2STModules), "eg2")
	if err != nil {
		t.Fatal(err)
	}
	if c.LayerNormEps != 1e-6 {
		t.Errorf("eps = %g, want Gemma's 1e-6 when rms_norm_eps is missing", c.LayerNormEps)
	}
	c.MaxPositionEmbeddings = 262144
	raw, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.MaxSeqLen() != eg2MaxSeqLen {
		t.Errorf("manifest MaxSeqLen = %d, want the %d cap", m.MaxSeqLen(), eg2MaxSeqLen)
	}
}
