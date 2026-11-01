// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// eg2MaxSeqLen caps EmbeddingGemma 2's sequence length at the 8192-token
// context the model card states. config.json carries max_position_embeddings
// 262144 (the Gemma 4 backbone's RoPE horizon) and tokenizer_config an
// effectively unlimited model_max_length; neither is the trained context.
const eg2MaxSeqLen = 8192

// eg2TextConfig is the text sub-config of an embedding_gemma2 checkpoint
// (config.json "text_config", or the whole file for a text-only export with
// model_type "embedding_gemma2_text").
type eg2TextConfig struct {
	ModelType             string   `json:"model_type"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	HeadDim               int      `json:"head_dim"`
	VocabSize             int      `json:"vocab_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	RMSNormEps            *float32 `json:"rms_norm_eps"`
	HiddenActivation      string   `json:"hidden_activation"`
	AttentionBias         bool     `json:"attention_bias"`
	SlidingWindow         int      `json:"sliding_window"`
	LayerTypes            []string `json:"layer_types"`
	PerLayerInputDim      int      `json:"hidden_size_per_layer_input"`
	EmbeddingDim          int      `json:"embedding_dim"`
	PerLayerConfig        map[string]struct {
		HeadDim          *int `json:"head_dim"`
		NumKeyValueHeads *int `json:"num_key_value_heads"`
	} `json:"per_layer_config"`
	RopeParameters map[string]struct {
		RopeTheta float64 `json:"rope_theta"`
		RopeType  string  `json:"rope_type"`
	} `json:"rope_parameters"`
	AttnLogitSoftcapping  *float64 `json:"attn_logit_softcapping"`
	FinalLogitSoftcapping *float64 `json:"final_logit_softcapping"`
	// Gemma 4 knobs this path does not implement: refused when set to a
	// non-default value, so a future variant fails loudly instead of
	// embedding wrong.
	PartialRotaryFactor       *float64 `json:"partial_rotary_factor"`
	NumKVSharedLayers         *int     `json:"num_kv_shared_layers"`
	UseBidirectionalAttention *bool    `json:"use_bidirectional_attention"`
}

// deriveEmbeddingGemma2 builds the Config for google/embeddinggemma-2's text
// path. Only text inputs are embedded; the vision and audio towers in the
// same checkpoint are ignored (never loaded). Every knob the forward pass
// depends on is read from config and anything rembed does not implement is
// refused, mirroring the other architectures' guards.
func deriveEmbeddingGemma2(dir, name string) (Config, error) {
	var c Config
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return c, err
	}
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return c, fmt.Errorf("model dir %s: config.json: %w", dir, err)
	}
	textRaw := raw
	if top.ModelType == "embedding_gemma2" {
		if len(top.TextConfig) == 0 {
			return c, fmt.Errorf("model dir %s: embedding_gemma2 config.json lacks text_config", dir)
		}
		textRaw = top.TextConfig
	}
	var t eg2TextConfig
	if err := json.Unmarshal(textRaw, &t); err != nil {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 text_config: %w", dir, err)
	}
	if t.ModelType != "" && t.ModelType != "embedding_gemma2_text" {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 text_config model_type=%q", dir, t.ModelType)
	}
	if t.HiddenActivation != "gelu_pytorch_tanh" {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 hidden_activation=%q — only gelu_pytorch_tanh is supported", dir, t.HiddenActivation)
	}
	if t.AttentionBias {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 with attention_bias is not supported (rembed's path is bias-free)", dir)
	}
	if (t.AttnLogitSoftcapping != nil && *t.AttnLogitSoftcapping != 0) || (t.FinalLogitSoftcapping != nil && *t.FinalLogitSoftcapping != 0) {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 logit softcapping is not supported", dir)
	}
	if t.PartialRotaryFactor != nil && *t.PartialRotaryFactor != 1 {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 partial_rotary_factor=%g is not supported (only full RoPE)", dir, *t.PartialRotaryFactor)
	}
	if t.NumKVSharedLayers != nil && *t.NumKVSharedLayers != 0 {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 num_kv_shared_layers=%d is not supported (no kv sharing)", dir, *t.NumKVSharedLayers)
	}
	if t.UseBidirectionalAttention != nil && !*t.UseBidirectionalAttention {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 use_bidirectional_attention=false is not supported (the embedder is bidirectional)", dir)
	}
	if len(t.LayerTypes) != t.NumHiddenLayers {
		return c, fmt.Errorf("model dir %s: embedding_gemma2 layer_types has %d entries for %d layers", dir, len(t.LayerTypes), t.NumHiddenLayers)
	}
	var thetas [2]float64 // [0] sliding, [1] full
	for i, kind := range []string{"sliding_attention", "full_attention"} {
		rp, ok := t.RopeParameters[kind]
		if !ok {
			return c, fmt.Errorf("model dir %s: embedding_gemma2 rope_parameters lacks %q", dir, kind)
		}
		if rp.RopeType != "" && rp.RopeType != "default" {
			return c, fmt.Errorf("model dir %s: embedding_gemma2 %s rope_type=%q is not supported (only default RoPE)", dir, kind, rp.RopeType)
		}
		thetas[i] = rp.RopeTheta
	}

	n := t.NumHiddenLayers
	global := make([]bool, n)
	headDim := make([]int, n)
	kvHeads := make([]int, n)
	for i, kind := range t.LayerTypes {
		switch kind {
		case "sliding_attention":
		case "full_attention":
			global[i] = true
		default:
			return c, fmt.Errorf("model dir %s: embedding_gemma2 layer %d type %q is not supported", dir, i, kind)
		}
		headDim[i], kvHeads[i] = t.HeadDim, t.NumKeyValueHeads
	}
	seen := map[int]string{}
	for key, pl := range t.PerLayerConfig {
		i, err := strconv.Atoi(key)
		if err != nil || i < 0 || i >= n {
			return c, fmt.Errorf("model dir %s: embedding_gemma2 per_layer_config key %q is not a layer index", dir, key)
		}
		if prev, dup := seen[i]; dup {
			return c, fmt.Errorf("model dir %s: embedding_gemma2 per_layer_config names layer %d twice (%q and %q)", dir, i, prev, key)
		}
		seen[i] = key
		if pl.HeadDim != nil {
			headDim[i] = *pl.HeadDim
		}
		if pl.NumKeyValueHeads != nil {
			kvHeads[i] = *pl.NumKeyValueHeads
		}
	}

	var tok struct {
		DoLowerCase *bool `json:"do_lower_case"`
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		if err := json.Unmarshal(raw, &tok); err != nil {
			return c, fmt.Errorf("model dir %s: tokenizer_config.json: %w", dir, err)
		}
	}
	if tok.DoLowerCase != nil && *tok.DoLowerCase {
		return c, fmt.Errorf("model dir %s: do_lower_case=true on an embedding_gemma2 model — the Gemma tokenizer is case-preserving", dir)
	}
	mode, normalize, err := stPipeline(dir, false)
	if err != nil {
		return c, err
	}

	c = Config{
		Name:                  name,
		ModelType:             "embedding_gemma2",
		HiddenSize:            t.HiddenSize,
		NumHiddenLayers:       n,
		NumAttentionHeads:     t.NumAttentionHeads,
		IntermediateSize:      t.IntermediateSize,
		VocabSize:             t.VocabSize,
		MaxPositionEmbeddings: min(t.MaxPositionEmbeddings, eg2MaxSeqLen),
		Pooling:               mode,
		Normalize:             normalize,
		Tokenizer:             "gemma",
		ClsToken:              "<bos>",
		SepToken:              "<eos>",
		HeadDim:               t.HeadDim,
		NumKeyValueHeads:      t.NumKeyValueHeads,
		LocalRopeTheta:        thetas[0],
		GlobalRopeTheta:       thetas[1],
		SlidingWindow:         t.SlidingWindow,
		LayerGlobal:           global,
		LayerHeadDim:          headDim,
		LayerKVHeads:          kvHeads,
		PerLayerInputDim:      t.PerLayerInputDim,
		EmbeddingDim:          t.EmbeddingDim,
	}
	// Gemma's RMSNorm eps defaults to 1e-6 (HF), not the 1e-12 validate
	// fills in for BERT-family configs.
	c.LayerNormEps = 1e-6
	if t.RMSNormEps != nil {
		c.LayerNormEps = *t.RMSNormEps
	}
	if dim, ok := poolingDim(dir); ok && dim != c.EmbeddingDim {
		return c, fmt.Errorf("model dir %s: 1_Pooling embedding_dimension %d disagrees with embedding_dim %d", dir, dim, c.EmbeddingDim)
	}
	return c, validate(&c, dir)
}

// poolingDim reads 1_Pooling/config.json's embedding_dimension, if present.
func poolingDim(dir string) (int, bool) {
	raw, err := os.ReadFile(filepath.Join(dir, "1_Pooling", "config.json"))
	if err != nil {
		return 0, false
	}
	var p struct {
		EmbeddingDimension *int `json:"embedding_dimension"`
	}
	if json.Unmarshal(raw, &p) != nil || p.EmbeddingDimension == nil {
		return 0, false
	}
	return *p.EmbeddingDimension, true
}

// validateEmbeddingGemma2 checks the per-layer schedule and the extra
// dimensions an embedding_gemma2 Config must carry (manifest or derived).
func validateEmbeddingGemma2(c *Config, source string) error {
	n := c.NumHiddenLayers
	if len(c.LayerGlobal) != n || len(c.LayerHeadDim) != n || len(c.LayerKVHeads) != n {
		return fmt.Errorf("%s: embedding_gemma2 requires layer_global, layer_head_dim and layer_kv_heads for all %d layers", source, n)
	}
	// One head_dim per attention type: the RoPE table is built once per type.
	dhByType := [2]int{}
	for i := range n {
		dh, kv := c.LayerHeadDim[i], c.LayerKVHeads[i]
		if dh <= 0 || dh%2 != 0 {
			return fmt.Errorf("%s: embedding_gemma2 layer %d requires an even head_dim > 0 (found %d)", source, i, dh)
		}
		if kv <= 0 || c.NumAttentionHeads%kv != 0 {
			return fmt.Errorf("%s: embedding_gemma2 layer %d: num_attention_heads %d not divisible by num_key_value_heads %d", source, i, c.NumAttentionHeads, kv)
		}
		ty := b2i(c.LayerGlobal[i])
		if dhByType[ty] == 0 {
			dhByType[ty] = dh
		} else if dhByType[ty] != dh {
			return fmt.Errorf("%s: embedding_gemma2 layer %d head_dim %d differs from other layers of the same attention type (%d)", source, i, dh, dhByType[ty])
		}
	}
	if c.SlidingWindow < 1 {
		return fmt.Errorf("%s: embedding_gemma2 requires a positive sliding_window (found %d)", source, c.SlidingWindow)
	}
	if c.GlobalRopeTheta <= 0 || c.LocalRopeTheta <= 0 {
		return fmt.Errorf("%s: embedding_gemma2 requires positive full and sliding rope_theta (found %g and %g)", source, c.GlobalRopeTheta, c.LocalRopeTheta)
	}
	if c.PerLayerInputDim < 1 || c.EmbeddingDim < 1 {
		return fmt.Errorf("%s: embedding_gemma2 requires positive per_layer_input_dim and embedding_dim (found %d and %d)", source, c.PerLayerInputDim, c.EmbeddingDim)
	}
	if c.Pooling != "mean" {
		return fmt.Errorf("%s: embedding_gemma2 requires mean pooling (found %q)", source, c.Pooling)
	}
	// The trained context, whatever the manifest copied from config.json
	// (262144, the RoPE horizon): longer inputs are truncated, not run.
	c.MaxPositionEmbeddings = min(c.MaxPositionEmbeddings, eg2MaxSeqLen)
	return nil
}
