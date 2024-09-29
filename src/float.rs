use half::f16;
use num_traits::Float;
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A trait representing basic floating-point operations, with support for custom implementations like f16 and f32.
/// This trait also provides various utility methods for mathematical operations commonly found in floating-point arithmetic.
pub trait FloatLike:
    Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + DivAssign
    + MulAssign
    + SubAssign
    + Sum
    + Into<f32>
    + PartialOrd
    + Default
{
    /// Square root function.
    fn sqrt(self) -> Self;
    /// Raises to an integer power.
    fn powi(self, n: i32) -> Self;
    /// Raises to a floating-point power.
    fn powf(self, n: Self) -> Self;
    /// Returns the absolute value.
    fn abs(self) -> Self;
    /// Returns the zero value.
    fn zero() -> Self;
    /// Returns the one value.
    fn one() -> Self;
    /// Checks if the number is NaN.
    fn is_nan(self) -> bool;
    /// Checks if the number is infinite.
    fn is_infinite(self) -> bool;
    /// Checks if the number is finite.
    fn is_finite(self) -> bool;
    /// Checks if the number is a normal floating-point number.
    fn is_normal(self) -> bool;
    /// Returns the reciprocal.
    fn recip(self) -> Self;
    /// Returns the sign of the number.
    fn signum(self) -> Self;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> Self;
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self;
    /// Returns the nearest integer to a number.
    fn round(self) -> Self;
    /// Returns the integer part of the number.
    fn trunc(self) -> Self;
    /// Returns the fractional part of the number.
    fn fract(self) -> Self;
    /// Returns the minimum of two numbers.
    fn min(self, other: Self) -> Self;
    /// Returns the maximum of two numbers.
    fn max(self, other: Self) -> Self;
    /// Returns e^(self).
    fn exp(self) -> Self;
    /// Returns the natural logarithm.
    fn ln(self) -> Self;
    /// Returns the logarithm of the number with a specific base.
    fn log(self, base: Self) -> Self;
    /// Returns 2^(self).
    fn exp2(self) -> Self;
    /// Returns the base-2 logarithm.
    fn log2(self) -> Self;
    /// Returns the base-10 logarithm.
    fn log10(self) -> Self;
    /// Returns the sine of the number.
    fn sin(self) -> Self;
    /// Returns the cosine of the number.
    fn cos(self) -> Self;
    /// Returns the tangent of the number.
    fn tan(self) -> Self;
    /// Returns the arcsine of the number.
    fn asin(self) -> Self;
    /// Returns the arccosine of the number.
    fn acos(self) -> Self;
    /// Returns the arctangent of the number.
    fn atan(self) -> Self;
    /// Returns the Euclidean distance between two numbers.
    fn hypot(self, other: Self) -> Self;
    /// Compares two floating-point numbers with a given tolerance.
    fn float_eq(self, other: Self, rel: f32) -> bool;
    /// Converts from usize to FloatLike type.
    fn from_usize(n: usize) -> Self;
    /// Adds two numbers.
    fn add(self, other: Self) -> Self;
    /// Divides two numbers.
    fn div(self, other: Self) -> Self;
    /// Multiplies two numbers.
    fn mul(self, other: Self) -> Self;
    /// Subtracts two numbers.
    fn sub(self, other: Self) -> Self;
    /// Performs in-place addition.
    fn add_assign(&mut self, other: Self);
    /// Performs in-place subtraction.
    fn sub_assign(&mut self, other: Self);
    /// Performs in-place multiplication.
    fn mul_assign(&mut self, other: Self);
    /// Performs in-place division.
    fn div_assign(&mut self, other: Self);
    /// Returns the sine and cosine of the number.
    fn sin_cos(self) -> (Self, Self);
    /// Returns negative infinity.
    fn neg_infinity() -> Self;
    /// Converts from f32 to FloatLike type.
    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

/// Implement FloatLike for f32.
impl FloatLike for f32 {
    // Each method directly maps to f32's built-in operations.
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }

    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        Float::powf(self, n)
    }

    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }

    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }

    fn is_normal(self) -> bool {
        Float::is_normal(self)
    }

    fn recip(self) -> Self {
        Float::recip(self)
    }

    fn signum(self) -> Self {
        Float::signum(self)
    }

    fn floor(self) -> Self {
        Float::floor(self)
    }

    fn ceil(self) -> Self {
        Float::ceil(self)
    }

    fn round(self) -> Self {
        Float::round(self)
    }

    fn trunc(self) -> Self {
        Float::trunc(self)
    }

    fn fract(self) -> Self {
        Float::fract(self)
    }

    fn min(self, other: Self) -> Self {
        Float::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        Float::max(self, other)
    }

    fn exp(self) -> Self {
        Float::exp(self)
    }

    fn ln(self) -> Self {
        Float::ln(self)
    }

    fn log(self, base: Self) -> Self {
        Float::log(self, base)
    }

    fn exp2(self) -> Self {
        Float::exp2(self)
    }

    fn log2(self) -> Self {
        Float::log2(self)
    }

    fn log10(self) -> Self {
        Float::log10(self)
    }

    fn sin(self) -> Self {
        Float::sin(self)
    }

    fn cos(self) -> Self {
        Float::cos(self)
    }

    fn tan(self) -> Self {
        Float::tan(self)
    }

    fn asin(self) -> Self {
        Float::asin(self)
    }

    fn acos(self) -> Self {
        Float::acos(self)
    }

    fn atan(self) -> Self {
        Float::atan(self)
    }

    fn hypot(self, other: Self) -> Self {
        Float::hypot(self, other)
    }

    fn float_eq(self, other: Self, rel: f32) -> bool {
        (self - other).abs() <= rel * (self.abs() + other.abs()) / 2.0
    }

    fn from_usize(n: usize) -> Self {
        n as f32
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn div(self, other: Self) -> Self {
        self / other
    }

    fn mul(self, other: Self) -> Self {
        self * other
    }

    fn sub(self, other: Self) -> Self {
        self - other
    }

    fn add_assign(&mut self, other: Self) {
        *self += other;
    }

    fn sub_assign(&mut self, other: Self) {
        *self -= other;
    }

    fn mul_assign(&mut self, other: Self) {
        *self *= other;
    }

    fn div_assign(&mut self, other: Self) {
        *self /= other;
    }

    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }

    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }

    fn from_f32(value: f32) -> Self {
        value
    }

    fn to_f32(self) -> f32 {
        self
    }
}

/// Implement FloatLike for f16.
/// This requires manual handling of conversions between f16 and f32 for mathematical operations.
impl FloatLike for f16 {
    fn sqrt(self) -> Self {
        f16::from_f32(f32::from(self).sqrt())
    }

    fn powi(self, n: i32) -> Self {
        f16::from_f32(f32::from(self).powi(n))
    }

    fn powf(self, n: Self) -> Self {
        f16::from_f32(f32::from(self).powf(f32::from(n)))
    }

    fn abs(self) -> Self {
        f16::from_f32(f32::from(self).abs())
    }

    fn zero() -> Self {
        f16::from_f32(0.0)
    }

    fn one() -> Self {
        f16::from_f32(1.0)
    }

    fn is_nan(self) -> bool {
        f32::from(self).is_nan()
    }

    fn is_infinite(self) -> bool {
        f32::from(self).is_infinite()
    }

    fn is_finite(self) -> bool {
        f32::from(self).is_finite()
    }

    fn is_normal(self) -> bool {
        f32::from(self).is_normal()
    }

    fn recip(self) -> Self {
        f16::from_f32(f32::from(self).recip())
    }

    fn signum(self) -> Self {
        f16::from_f32(f32::from(self).signum())
    }

    fn floor(self) -> Self {
        f16::from_f32(f32::from(self).floor())
    }

    fn ceil(self) -> Self {
        f16::from_f32(f32::from(self).ceil())
    }

    fn round(self) -> Self {
        f16::from_f32(f32::from(self).round())
    }

    fn trunc(self) -> Self {
        f16::from_f32(f32::from(self).trunc())
    }

    fn fract(self) -> Self {
        f16::from_f32(f32::from(self).fract())
    }

    fn min(self, other: Self) -> Self {
        f16::from_f32(f32::from(self).min(f32::from(other)))
    }

    fn max(self, other: Self) -> Self {
        f16::from_f32(f32::from(self).max(f32::from(other)))
    }

    fn exp(self) -> Self {
        f16::from_f32(f32::from(self).exp())
    }

    fn ln(self) -> Self {
        f16::from_f32(f32::from(self).ln())
    }

    fn log(self, base: Self) -> Self {
        f16::from_f32(f32::from(self).log(f32::from(base)))
    }

    fn exp2(self) -> Self {
        f16::from_f32(f32::from(self).exp2())
    }

    fn log2(self) -> Self {
        f16::from_f32(f32::from(self).log2())
    }

    fn log10(self) -> Self {
        f16::from_f32(f32::from(self).log10())
    }

    fn sin(self) -> Self {
        f16::from_f32(f32::from(self).sin())
    }

    fn cos(self) -> Self {
        f16::from_f32(f32::from(self).cos())
    }

    fn tan(self) -> Self {
        f16::from_f32(f32::from(self).tan())
    }

    fn asin(self) -> Self {
        f16::from_f32(f32::from(self).asin())
    }

    fn acos(self) -> Self {
        f16::from_f32(f32::from(self).acos())
    }

    fn atan(self) -> Self {
        f16::from_f32(f32::from(self).atan())
    }

    fn hypot(self, other: Self) -> Self {
        f16::from_f32(f32::from(self).hypot(f32::from(other)))
    }

    fn float_eq(self, other: Self, rel: f32) -> bool {
        let x_f32: f32 = self.into();
        let y_f32: f32 = other.into();
        (x_f32 - y_f32).abs() <= rel * (x_f32.abs() + y_f32.abs()) / 2.0
    }

    fn from_usize(n: usize) -> Self {
        f16::from_f32(n as f32)
    }

    fn add(self, other: Self) -> Self {
        f16::from_f32(f32::from(self) + f32::from(other))
    }

    fn div(self, other: Self) -> Self {
        f16::from_f32(f32::from(self) / f32::from(other))
    }

    fn mul(self, other: Self) -> Self {
        f16::from_f32(f32::from(self) * f32::from(other))
    }

    fn sub(self, other: Self) -> Self {
        f16::from_f32(f32::from(self) - f32::from(other))
    }

    fn add_assign(&mut self, other: Self) {
        *self = f16::from_f32(f32::from(*self) + f32::from(other));
    }

    fn sub_assign(&mut self, other: Self) {
        *self = f16::from_f32(f32::from(*self) - f32::from(other));
    }

    fn mul_assign(&mut self, other: Self) {
        *self = f16::from_f32(f32::from(*self) * f32::from(other));
    }

    fn div_assign(&mut self, other: Self) {
        *self = f16::from_f32(f32::from(*self) / f32::from(other));
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = f32::from(self).sin_cos();
        (f16::from_f32(sin), f16::from_f32(cos))
    }

    fn neg_infinity() -> Self {
        f16::from_f32(f32::NEG_INFINITY)
    }

    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

// Tests for the FloatLike trait.
#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_sqrt() {
        let x_f32: f32 = 4.0;
        let x_f16 = f16::from_f32(4.0);

        assert_eq!(x_f32.sqrt(), 2.0);
        assert_eq!(f32::from(x_f16.sqrt()), 2.0);
    }

    #[test]
    fn test_powi() {
        let x_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32.powi(3), 8.0);
        assert_eq!(f32::from(x_f16.powi(3)), 8.0);
    }

    #[test]
    fn test_powf() {
        let x_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32.powf(3.0), 8.0);
        assert_eq!(f32::from(x_f16.powf(f16::from_f32(3.0))), 8.0);
    }

    #[test]
    fn test_abs() {
        let x_f32: f32 = -3.0;
        let x_f16 = f16::from_f32(-3.0);

        assert_eq!(x_f32.abs(), 3.0);
        assert_eq!(f32::from(x_f16.abs()), 3.0);
    }

    #[test]
    fn test_zero() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::from(f16::zero()), 0.0);
    }

    #[test]
    fn test_one() {
        assert_eq!(f32::one(), 1.0);
        assert_eq!(f32::from(f16::one()), 1.0);
    }

    #[test]
    fn test_is_nan() {
        let nan_f32: f32 = f32::NAN;
        let nan_f16 = f16::NAN;

        assert!(nan_f32.is_nan());
        assert!(nan_f16.is_nan());
    }

    #[test]
    fn test_is_infinite() {
        let inf_f32: f32 = f32::INFINITY;
        let inf_f16 = f16::INFINITY;

        assert!(inf_f32.is_infinite());
        assert!(inf_f16.is_infinite());
    }

    #[test]
    fn test_is_finite() {
        let x_f32: f32 = 5.0;
        let x_f16 = f16::from_f32(5.0);

        assert!(x_f32.is_finite());
        assert!(x_f16.is_finite());
    }

    #[test]
    fn test_is_normal() {
        let x_f32: f32 = 5.0;
        let x_f16 = f16::from_f32(5.0);

        assert!(x_f32.is_normal());
        assert!(x_f16.is_normal());
    }

    #[test]
    fn test_recip() {
        let x_f32: f32 = 4.0;
        let x_f16 = f16::from_f32(4.0);

        assert_eq!(x_f32.recip(), 0.25);
        assert_eq!(f32::from(x_f16.recip()), 0.25);
    }

    #[test]
    fn test_signum() {
        let x_f32: f32 = -3.0;
        let x_f16 = f16::from_f32(-3.0);

        assert_eq!(x_f32.signum(), -1.0);
        assert_eq!(f32::from(x_f16.signum()), -1.0);
    }

    #[test]
    fn test_floor() {
        let x_f32: f32 = 3.7;
        let x_f16 = f16::from_f32(3.7);

        assert_eq!(x_f32.floor(), 3.0);
        assert_eq!(f32::from(x_f16.floor()), 3.0);
    }

    #[test]
    fn test_ceil() {
        let x_f32: f32 = 3.2;
        let x_f16 = f16::from_f32(3.2);

        assert_eq!(x_f32.ceil(), 4.0);
        assert_eq!(f32::from(x_f16.ceil()), 4.0);
    }

    #[test]
    fn test_round() {
        let x_f32: f32 = 3.5;
        let x_f16 = f16::from_f32(3.5);

        assert_eq!(x_f32.round(), 4.0);
        assert_eq!(f32::from(x_f16.round()), 4.0);
    }

    #[test]
    fn test_trunc() {
        let x_f32: f32 = 3.9;
        let x_f16 = f16::from_f32(3.9);

        assert_eq!(x_f32.trunc(), 3.0);
        assert_eq!(f32::from(x_f16.trunc()), 3.0);
    }

    #[test]
    fn test_fract() {
        let x_f32: f32 = 3.75;
        let x_f16 = f16::from_f32(3.75);

        assert_eq!(x_f32.fract(), 0.75);
        assert_eq!(f32::from(x_f16.fract()), 0.75);
    }

    #[test]
    fn test_min() {
        let x_f32: f32 = 3.0;
        let y_f32: f32 = 4.0;

        let x_f16 = f16::from_f32(3.0);
        let y_f16 = f16::from_f32(4.0);

        assert_eq!(x_f32.min(y_f32), 3.0);
        assert_eq!(f32::from(x_f16.min(y_f16)), 3.0);
    }

    #[test]
    fn test_max() {
        let x_f32: f32 = 3.0;
        let y_f32: f32 = 4.0;

        let x_f16 = f16::from_f32(3.0);
        let y_f16 = f16::from_f32(4.0);

        assert_eq!(x_f32.max(y_f32), 4.0);
        assert_eq!(f32::from(x_f16.max(y_f16)), 4.0);
    }

    #[test]
    fn test_exp() {
        let x_f32: f32 = 1.0;
        let x_f16 = f16::from_f32(1.0);

        assert_eq!(x_f32.exp(), 2.7182817);
        assert!(x_f16.exp().float_eq(f16::from_f32(2.7182817), 0.0001));
    }

    #[test]
    fn test_ln() {
        let x_f32: f32 = 2.7182817;
        let x_f16 = f16::from_f32(2.7182817);

        assert!(x_f32.ln().float_eq(1.0, 0.000001));
        assert!(x_f16.ln().float_eq(f16::from_f32(1.0), 0.0001));
    }

    #[test]
    fn test_log() {
        let x_f32: f32 = 100.0;
        let base_f32: f32 = 10.0;

        let x_f16 = f16::from_f32(100.0);
        let base_f16 = f16::from_f32(10.0);

        assert_eq!(x_f32.log(base_f32), 2.0);
        assert_eq!(f32::from(x_f16.log(base_f16)), 2.0);
    }

    #[test]
    fn test_exp2() {
        let x_f32: f32 = 3.0;
        let x_f16 = f16::from_f32(3.0);

        assert_eq!(x_f32.exp2(), 8.0);
        assert_eq!(f32::from(x_f16.exp2()), 8.0);
    }

    #[test]
    fn test_log2() {
        let x_f32: f32 = 8.0;
        let x_f16 = f16::from_f32(8.0);

        assert_eq!(x_f32.log2(), 3.0);
        assert_eq!(f32::from(x_f16.log2()), 3.0);
    }

    #[test]
    fn test_log10() {
        let x_f32: f32 = 100.0;
        let x_f16 = f16::from_f32(100.0);

        assert_eq!(x_f32.log10(), 2.0);
        assert_eq!(f32::from(x_f16.log10()), 2.0);
    }

    #[test]
    fn test_sin() {
        let x_f32: f32 = 1.0;
        let x_f16 = f16::from_f32(1.0);

        assert_eq!(x_f32.sin(), 0.84147096);
        assert!(x_f16.sin().float_eq(f16::from_f32(0.84147096), 0.0001));
    }

    #[test]
    fn test_cos() {
        let x_f32: f32 = 1.0;
        let x_f16 = f16::from_f32(1.0);

        assert_eq!(x_f32.cos(), 0.5403023);
        assert!(x_f16.cos().float_eq(f16::from_f32(0.5403023), 0.0001));
    }

    #[test]
    fn test_tan() {
        let x_f32: f32 = 1.0;
        let x_f16 = f16::from_f32(1.0);

        assert_eq!(x_f32.tan(), 1.5574077);
        assert!(x_f16.tan().float_eq(f16::from_f32(1.5574077), 0.0001));
    }

    #[test]
    fn test_asin() {
        let x_f32: f32 = 0.5;
        let x_f16 = f16::from_f32(0.5);

        assert_eq!(x_f32.asin(), 0.5235988);
        assert!(x_f16.asin().float_eq(f16::from_f32(0.5235988), 0.0001));
    }

    #[test]
    fn test_acos() {
        let x_f32: f32 = 0.5;
        let x_f16 = f16::from_f32(0.5);

        assert_eq!(x_f32.acos(), 1.0471976);
        assert!(x_f16.acos().float_eq(f16::from_f32(1.0471976), 0.0001))
    }

    #[test]
    fn test_atan() {
        let x_f32: f32 = 0.5;
        let x_f16 = f16::from_f32(0.5);

        assert_eq!(x_f32.atan(), 0.4636476);
        assert!(x_f16.atan().float_eq(f16::from_f32(0.4636476), 0.0001));
    }

    #[test]
    fn test_hypot() {
        let x_f32: f32 = 3.0;
        let y_f32: f32 = 4.0;

        let x_f16 = f16::from_f32(3.0);
        let y_f16 = f16::from_f32(4.0);

        assert_eq!(x_f32.hypot(y_f32), 5.0);
        assert_eq!(f32::from(x_f16.hypot(y_f16)), 5.0);
    }

    #[test]
    fn test_float_eq() {
        let x_f32: f32 = 1.00001;
        let y_f32: f32 = 1.00002;

        let x_f16 = f16::from_f32(1.00001);
        let y_f16 = f16::from_f32(1.00002);

        assert!(x_f32.float_eq(y_f32, 0.0001));
        assert!(x_f16.float_eq(y_f16, 0.0001));
    }

    #[test]
    fn test_from_usize() {
        let x: usize = 10;

        let y_f32 = f32::from_usize(x);
        let y_f16 = f16::from_usize(x);

        assert_eq!(y_f32, 10.0);
        assert_eq!(f32::from(y_f16), 10.0);
    }

    #[test]
    fn test_add() {
        let x_f32: f32 = 3.0;
        let y_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(3.0);
        let y_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32 + y_f32, 5.0);
        assert_eq!(f32::from(x_f16 + y_f16), 5.0);
    }

    #[test]
    fn test_div() {
        let x_f32: f32 = 6.0;
        let y_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(6.0);
        let y_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32 / y_f32, 3.0);
        assert_eq!(f32::from(x_f16 / y_f16), 3.0);
    }

    #[test]
    fn test_mul() {
        let x_f32: f32 = 3.0;
        let y_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(3.0);
        let y_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32 * y_f32, 6.0);
        assert_eq!(f32::from(x_f16 * y_f16), 6.0);
    }

    #[test]
    fn test_sub() {
        let x_f32: f32 = 5.0;
        let y_f32: f32 = 2.0;
        let x_f16 = f16::from_f32(5.0);
        let y_f16 = f16::from_f32(2.0);

        assert_eq!(x_f32 - y_f32, 3.0);
        assert_eq!(f32::from(x_f16 - y_f16), 3.0);
    }

    #[test]
    fn test_add_assign() {
        let mut x_f32: f32 = 2.0;
        let y_f32: f32 = 3.0;
        x_f32 += y_f32;
        assert_eq!(x_f32, 5.0);

        let mut x_f16 = f16::from_f32(2.0);
        let y_f16 = f16::from_f32(3.0);
        x_f16 += y_f16;
        assert_eq!(f32::from(x_f16), 5.0);
    }

    #[test]
    fn test_sub_assign() {
        let mut x_f32: f32 = 5.0;
        let y_f32: f32 = 3.0;
        x_f32 -= y_f32;
        assert_eq!(x_f32, 2.0);

        let mut x_f16 = f16::from_f32(5.0);
        let y_f16 = f16::from_f32(3.0);
        x_f16 -= y_f16;
        assert_eq!(f32::from(x_f16), 2.0);
    }

    #[test]
    fn test_mul_assign() {
        let mut x_f32: f32 = 2.0;
        let y_f32: f32 = 3.0;
        x_f32 *= y_f32;
        assert_eq!(x_f32, 6.0);

        let mut x_f16 = f16::from_f32(2.0);
        let y_f16 = f16::from_f32(3.0);
        x_f16 *= y_f16;
        assert_eq!(f32::from(x_f16), 6.0);
    }

    #[test]
    fn test_div_assign() {
        let mut x_f32: f32 = 6.0;
        let y_f32: f32 = 3.0;
        x_f32 /= y_f32;
        assert_eq!(x_f32, 2.0);

        let mut x_f16 = f16::from_f32(6.0);
        let y_f16 = f16::from_f32(3.0);
        x_f16 /= y_f16;
        assert_eq!(f32::from(x_f16), 2.0);
    }

    #[test]
    fn test_sin_cos() {
        let x_f32: f32 = 1.0;
        let x_f16 = f16::from_f32(1.0);

        let (sin_f32, cos_f32) = x_f32.sin_cos();
        let (sin_f16, cos_f16) = x_f16.sin_cos();

        assert!(sin_f32.float_eq(0.84147096, 0.0001));
        assert!(cos_f32.float_eq(0.5403023, 0.0001));

        assert!(sin_f16.float_eq(f16::from_f32(0.84147096), 0.0001));
        assert!(cos_f16.float_eq(f16::from_f32(0.5403023), 0.0001));
    }

    #[test]
    fn test_neg_infinity() {
        assert_eq!(<f32 as FloatLike>::neg_infinity(), f32::NEG_INFINITY);
        assert_eq!(f32::from(f16::neg_infinity()), f32::NEG_INFINITY);
    }

    #[test]
    fn test_sum() {
        let vec_f32: Vec<f32> = vec![1.0, 2.0, 3.0];
        let vec_f16: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];

        let sum_f32: f32 = vec_f32.iter().cloned().sum();
        let sum_f16: f16 = vec_f16.iter().cloned().sum();

        assert_eq!(sum_f32, 6.0);
        assert_eq!(f32::from(sum_f16), 6.0);
    }
}
