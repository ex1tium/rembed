// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"math"
	"testing"
)

// TestExpf32AgainstFloat64 pins the fast exp at its real, measured bounds:
// ~2e-7 relative near zero, degrading with |x| as the two-part range
// reduction cancels (~4e-6 at |x|≈85). The review that corrected an
// optimistic "~2e-7 everywhere" claim is why both tiers are pinned.
func TestExpf32AgainstFloat64(t *testing.T) {
	for x := float32(-87); x <= 88; x += 0.001 {
		got := float64(expf32(x))
		want := math.Exp(float64(x))
		bound := 5e-6
		if x > -20 && x < 20 {
			bound = 5e-7
		}
		if rel := math.Abs(got-want) / want; rel > bound {
			t.Fatalf("expf32(%v)=%g want %g (rel %g > %g)", x, got, want, rel, bound)
		}
	}
	// The (88.03, 88.72] region has finite float32 results even though the
	// exponent reassembly needs 2^128 — the split-scale path must not
	// return +Inf there.
	for _, x := range []float32{88.1, 88.5, 88.7} {
		if v := float64(expf32(x)); math.IsInf(v, 1) {
			t.Fatalf("expf32(%v)=+Inf, want finite", x)
		}
	}
	// Clamp regions.
	if v := expf32(-100); v != 0 {
		t.Fatalf("expf32(-100)=%v want 0", v)
	}
	if v := expf32(200); !math.IsInf(float64(v), 1) {
		t.Fatalf("expf32(200)=%v want +Inf", v)
	}
	if v := expf32(0); v != 1 {
		t.Fatalf("expf32(0)=%v want 1", v)
	}
}

// TestErff32AgainstFloat64 pins the fast erf to ~5e-7 absolute error.
func TestErff32AgainstFloat64(t *testing.T) {
	for x := float32(-10); x <= 10; x += 0.001 {
		got := float64(erff32(x))
		want := math.Erf(float64(x))
		if d := math.Abs(got - want); d > 5e-7 {
			t.Fatalf("erff32(%v)=%g want %g (diff %g)", x, got, want, d)
		}
	}
	if v := erff32(0); v != 0 {
		t.Fatalf("erff32(0)=%v want 0", v)
	}
}

func BenchmarkStdlibExp(b *testing.B) {
	var s float64
	x := 0.0
	for b.Loop() {
		s += math.Exp(x)
		x -= 1e-6
	}
	_ = s
}

func BenchmarkExpf32(b *testing.B) {
	var s float32
	x := float32(0)
	for b.Loop() {
		s += expf32(x)
		x -= 1e-6
	}
	_ = s
}

func BenchmarkStdlibErf(b *testing.B) {
	var s float64
	x := 0.0
	for b.Loop() {
		s += math.Erf(x)
		x += 1e-6
	}
	_ = s
}

func BenchmarkErff32(b *testing.B) {
	var s float32
	x := float32(0)
	for b.Loop() {
		s += erff32(x)
		x += 1e-6
	}
	_ = s
}

// TestGELUTanhAgainstFloat64 pins the float32 σ(2u) form of GELUTanh to the
// float64 0.5·x·(1+tanh(u)) definition across the range activations reach,
// including the far tails where one branch underflows to x or 0.
func TestGELUTanhAgainstFloat64(t *testing.T) {
	var xs []float32
	for x := float32(-30); x <= 30; x += 0.0005 {
		xs = append(xs, x)
	}
	xs = append(xs, 0, -1e-30, 1e-30, -100, 100)
	got := append([]float32(nil), xs...)
	GELUTanh(got)
	var worst float64
	for i, v := range xs {
		v64 := float64(v)
		want := 0.5 * v64 * (1 + math.Tanh(0.7978845608028654*(v64+0.044715*v64*v64*v64)))
		// Relative to max(|want|, 1): absolute near zero, relative in the tails.
		d := math.Abs(float64(got[i])-want) / math.Max(math.Abs(want), 1)
		worst = math.Max(worst, d)
		if d > 1e-6 || math.IsNaN(float64(got[i])) {
			t.Fatalf("GELUTanh(%v)=%v want %v (err %g > 1e-6)", v, got[i], want, d)
		}
	}
	t.Logf("worst scaled error %.3g", worst)
}

// TestGELUTanhEdgeCases pins the special values: -0 stays -0, NaN passes
// through, +Inf stays +Inf, and -Inf and huge negatives go to (signed) zero
// without overflow.
func TestGELUTanhEdgeCases(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	x := []float32{negZero, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), -1e30, 1e30}
	GELUTanh(x)
	if x[0] != 0 || !math.Signbit(float64(x[0])) {
		t.Errorf("GELUTanh(-0) = %v, want -0", x[0])
	}
	if !math.IsNaN(float64(x[1])) {
		t.Errorf("GELUTanh(NaN) = %v, want NaN", x[1])
	}
	if !math.IsInf(float64(x[2]), 1) {
		t.Errorf("GELUTanh(+Inf) = %v, want +Inf", x[2])
	}
	if math.IsNaN(float64(x[3])) || math.Abs(float64(x[3])) > 1e-30 {
		t.Errorf("GELUTanh(-Inf) = %v, want ~0", x[3])
	}
	if math.IsNaN(float64(x[4])) || math.Abs(float64(x[4])) > 1e-6 || x[5] != 1e30 {
		t.Errorf("GELUTanh(-1e30, 1e30) = %v, %v", x[4], x[5])
	}
}
