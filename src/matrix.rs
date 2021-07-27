use crate::tuple::Tuple;
use approx::relative_eq;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use itertools::iproduct;
use std::ops::{Index, IndexMut, Mul};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<const M: usize, const N: usize> {
    data: [[f32; N]; M],
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn new(data: [[f32; N]; M]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self {
            data: [[0.0; N]; M],
        }
    }

    pub fn nrows(&self) -> usize {
        M
    }

    pub fn ncols(&self) -> usize {
        N
    }

    pub fn transpose(&self) -> Matrix<N, M> {
        let mut res = Matrix::zero();
        for (i, j) in iproduct!(0..M, 0..N) {
            res[i][j] = self[j][i];
        }
        res
    }
}

impl<const N: usize> Matrix<N, N> {
    pub fn identity() -> Self {
        let mut res = Matrix::zero();
        for i in 0..N {
            res[i][i] = 1.0;
        }
        res
    }
}

impl Matrix<2, 2> {
    pub fn determinant(&self) -> f32 {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }
}

// TODO: Re-implement these using const generic expressions.
impl Matrix<3, 3> {
    pub fn submatrix(&self, row: usize, col: usize) -> Matrix<2, 2> {
        let mut res = Matrix::zero();
        for (i, j) in iproduct!(0..2, 0..2) {
            let iskip = (i >= row) as usize;
            let jskip = (j >= col) as usize;
            res[i][j] = self[i + iskip][j + jskip];
        }
        res
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 == 0 {
            self.minor(row, col)
        } else {
            -self.minor(row, col)
        }
    }

    pub fn determinant(&self) -> f32 {
        (0..3).map(|col| self[0][col] * self.cofactor(0, col)).sum()
    }

    pub fn invertible(&self) -> bool {
        relative_eq!(self.determinant(), 0.0)
    }

    pub fn inverse(&self) -> Self {
        let mut res = Matrix::zero();
        let det = self.determinant();
        for (i, j) in iproduct!(0..3, 0..3) {
            res[j][i] = self.cofactor(i, j) / det;
        }
        res
    }
}

impl Matrix<4, 4> {
    pub fn submatrix(&self, row: usize, col: usize) -> Matrix<3, 3> {
        let mut res = Matrix::zero();
        for (i, j) in iproduct!(0..3, 0..3) {
            let iskip = (i >= row) as usize;
            let jskip = (j >= col) as usize;
            res[i][j] = self[i + iskip][j + jskip];
        }
        res
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        self.submatrix(row, col).determinant()
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f32 {
        if (row + col) % 2 == 0 {
            self.minor(row, col)
        } else {
            -self.minor(row, col)
        }
    }

    pub fn determinant(&self) -> f32 {
        (0..4).map(|col| self[0][col] * self.cofactor(0, col)).sum()
    }

    pub fn invertible(&self) -> bool {
        relative_eq!(self.determinant(), 0.0)
    }

    pub fn inverse(&self) -> Self {
        let mut res = Matrix::zero();
        let det = self.determinant();
        for (i, j) in iproduct!(0..4, 0..4) {
            res[j][i] = self.cofactor(i, j) / det;
        }
        res
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<N, P>>
    for Matrix<M, N>
{
    type Output = Matrix<M, P>;

    fn mul(self, rhs: Matrix<N, P>) -> Matrix<M, P> {
        let mut res = Matrix::zero();
        for (i, j, k) in iproduct!(0..M, 0..P, 0..N) {
            res[i][j] += self[i][k] * rhs[k][j];
        }
        res
    }
}

impl Mul<Tuple> for Matrix<4, 4> {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Tuple {
        Tuple {
            x: rhs.dot(Tuple::from(self[0])),
            y: rhs.dot(Tuple::from(self[1])),
            z: rhs.dot(Tuple::from(self[2])),
            w: rhs.dot(Tuple::from(self[3])),
        }
    }
}

impl Mul<Matrix<4, 4>> for Tuple {
    type Output = Tuple;

    fn mul(self, rhs: Matrix<4, 4>) -> Tuple {
        let cols = rhs.transpose();
        Tuple {
            x: self.dot(Tuple::from(cols[0])),
            y: self.dot(Tuple::from(cols[1])),
            z: self.dot(Tuple::from(cols[2])),
            w: self.dot(Tuple::from(cols[3])),
        }
    }
}

impl<const M: usize, const N: usize> Index<usize> for Matrix<M, N> {
    type Output = [f32; N];

    fn index(&self, row: usize) -> &[f32; N] {
        &self.data[row]
    }
}

impl<const M: usize, const N: usize> IndexMut<usize> for Matrix<M, N> {
    fn index_mut(&mut self, row: usize) -> &mut [f32; N] {
        &mut self.data[row]
    }
}

impl<const M: usize, const N: usize> AbsDiffEq for Matrix<M, N> {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        iproduct!(0..M, 0..N).all(|(row, col)| {
            self.data[row][col].abs_diff_eq(&other.data[row][col], epsilon)
        })
    }
}

impl<const M: usize, const N: usize> RelativeEq for Matrix<M, N> {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        iproduct!(0..M, 0..N).all(|(row, col)| {
            self.data[row][col].relative_eq(
                &other.data[row][col],
                epsilon,
                max_relative,
            )
        })
    }
}

impl<const M: usize, const N: usize> UlpsEq for Matrix<M, N> {
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }

    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_ulps: u32,
    ) -> bool {
        iproduct!(0..M, 0..N).all(|(row, col)| {
            self.data[row][col].ulps_eq(
                &other.data[row][col],
                epsilon,
                max_ulps,
            )
        })
    }
}

#[cfg(test)]
mod tests_matrix {
    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};

    #[test]
    fn construct_matrix() {
        let m: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.5, 6.5, 7.5, 8.5],
            [9.0, 10.0, 11.0, 12.0],
            [13.5, 14.5, 15.5, 16.5],
        ]);
        assert_relative_eq!(m[0][0], 1.0);
        assert_relative_eq!(m[0][3], 4.0);
        assert_relative_eq!(m[1][0], 5.5);
        assert_relative_eq!(m[1][2], 7.5);
        assert_relative_eq!(m[2][2], 11.0);
        assert_relative_eq!(m[3][0], 13.5);
        assert_relative_eq!(m[3][2], 15.5);

        let m: Matrix<2, 2> = Matrix::new([[-3.0, 5.0], [1.0, -2.0]]);
        assert_relative_eq!(m[0][0], -3.0);
        assert_relative_eq!(m[0][1], 5.0);
        assert_relative_eq!(m[1][0], 1.0);
        assert_relative_eq!(m[1][1], -2.0);

        let m: Matrix<3, 3> = Matrix::new([
            [-3.0, 5.0, 0.0],
            [1.0, -2.0, -7.0],
            [0.0, 1.0, 1.0],
        ]);
        assert_relative_eq!(m[0][0], -3.0);
        assert_relative_eq!(m[1][1], -2.0);
        assert_relative_eq!(m[2][2], 1.0);
    }

    #[test]
    fn matrix_equality() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0],
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ]);
        assert_relative_eq!(a, a);
        assert_relative_ne!(a, b);
    }

    #[test]
    fn matrix_mult() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [-2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, -1.0],
            [4.0, 3.0, 6.0, 5.0],
            [1.0, 2.0, 7.0, 8.0],
        ]);
        assert_relative_eq!(
            a * b,
            Matrix::new([
                [20.0, 22.0, 50.0, 48.0],
                [44.0, 54.0, 114.0, 108.0],
                [40.0, 58.0, 110.0, 102.0],
                [16.0, 26.0, 46.0, 42.0],
            ])
        );
    }

    #[test]
    fn matrix_mult_tuple() {
        let a: Matrix<4, 4> = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 4.0, 2.0],
            [8.0, 6.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let b: Tuple = Tuple::new(1.0, 2.0, 3.0, 1.0);
        assert_relative_eq!(a * b, Tuple::new(18.0, 24.0, 33.0, 1.0));
    }

    #[test]
    fn matrix_mult_identity() {
        let a: Matrix<4, 4> = Matrix::new([
            [0.0, 1.0, 2.0, 4.0],
            [1.0, 2.0, 4.0, 8.0],
            [2.0, 4.0, 8.0, 16.0],
            [4.0, 8.0, 16.0, 32.0],
        ]);
        assert_relative_eq!(a * Matrix::identity(), a);

        let a: Tuple = Tuple::new(1.0, 2.0, 3.0, 4.0);
        assert_relative_eq!(Matrix::identity() * a, a);
    }

    #[test]
    fn matrix_transpose() {
        let a: Matrix<4, 4> = Matrix::new([
            [0.0, 9.0, 3.0, 0.0],
            [9.0, 8.0, 0.0, 8.0],
            [1.0, 8.0, 5.0, 3.0],
            [0.0, 0.0, 5.0, 8.0],
        ]);
        assert_relative_eq!(
            a.transpose(),
            Matrix::new([
                [0.0, 9.0, 1.0, 0.0],
                [9.0, 8.0, 8.0, 0.0],
                [3.0, 0.0, 5.0, 5.0],
                [0.0, 8.0, 3.0, 8.0],
            ])
        );

        let a: Matrix<4, 4> = Matrix::identity().transpose();
        assert_relative_eq!(a, Matrix::identity());
    }

    #[test]
    fn matrix_determinant() {
        let a: Matrix<2, 2> = Matrix::new([[1.0, 5.0], [-3.0, 2.0]]);
        assert_relative_eq!(a.determinant(), 17.0);

        let a: Matrix<3, 3> =
            Matrix::new([[1.0, 2.0, 6.0], [-5.0, 8.0, -4.0], [2.0, 6.0, 4.0]]);
        assert_relative_eq!(a.cofactor(0, 0), 56.0);
        assert_relative_eq!(a.cofactor(0, 1), 12.0);
        assert_relative_eq!(a.cofactor(0, 2), -46.0);
        assert_relative_eq!(a.determinant(), -196.0);

        let a: Matrix<4, 4> = Matrix::new([
            [-2.0, -8.0, 3.0, 5.0],
            [-3.0, 1.0, 7.0, 3.0],
            [1.0, 2.0, -9.0, 6.0],
            [-6.0, 7.0, 7.0, -9.0],
        ]);
        assert_relative_eq!(a.cofactor(0, 0), 690.0);
        assert_relative_eq!(a.cofactor(0, 1), 447.0);
        assert_relative_eq!(a.cofactor(0, 2), 210.0);
        assert_relative_eq!(a.cofactor(0, 3), 51.0);
        assert_relative_eq!(a.determinant(), -4071.0);
    }

    #[test]
    fn matrix_submatrix() {
        let a: Matrix<3, 3> =
            Matrix::new([[1.0, 5.0, 0.0], [-3.0, 2.0, 7.0], [0.0, 6.0, -3.0]]);
        assert_relative_eq!(
            a.submatrix(0, 2),
            Matrix::new([[-3.0, 2.0], [0.0, 6.0]])
        );

        let a: Matrix<4, 4> = Matrix::new([
            [-6.0, 1.0, 1.0, 6.0],
            [-8.0, 5.0, 8.0, 6.0],
            [-1.0, 0.0, 8.0, 2.0],
            [-7.0, 1.0, -1.0, 1.0],
        ]);
        assert_relative_eq!(
            a.submatrix(2, 1),
            Matrix::new([
                [-6.0, 1.0, 6.0],
                [-8.0, 8.0, 6.0],
                [-7.0, -1.0, 1.0],
            ])
        );
    }

    #[test]
    fn matrix_minor() {
        let a: Matrix<3, 3> = Matrix::new([
            [3.0, 5.0, 0.0],
            [2.0, -1.0, -7.0],
            [6.0, -1.0, 5.0],
        ]);
        let b = a.submatrix(1, 0);
        assert_relative_eq!(b.determinant(), 25.0);
        assert_relative_eq!(a.minor(1, 0), 25.0);
    }

    #[test]
    fn matrix_cofactor() {
        let a: Matrix<3, 3> = Matrix::new([
            [3.0, 5.0, 0.0],
            [2.0, -1.0, -7.0],
            [6.0, -1.0, 5.0],
        ]);
        assert_relative_eq!(a.minor(0, 0), -12.0);
        assert_relative_eq!(a.cofactor(0, 0), -12.0);
        assert_relative_eq!(a.minor(1, 0), 25.0);
        assert_relative_eq!(a.cofactor(1, 0), -25.0);
    }

    #[test]
    fn matrix_invertibility() {
        let a: Matrix<4, 4> = Matrix::new([
            [6.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 7.0, 6.0],
            [4.0, -9.0, 3.0, -7.0],
            [9.0, 1.0, 7.0, -6.0],
        ]);
        assert_relative_eq!(a.determinant(), -2120.0);

        let a: Matrix<4, 4> = Matrix::new([
            [-4.0, 2.0, -2.0, -3.0],
            [9.0, 6.0, 2.0, 6.0],
            [0.0, -5.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        assert_relative_eq!(a.determinant(), 0.0);
    }

    #[test]
    fn matrix_inverse() {
        let a: Matrix<4, 4> = Matrix::new([
            [-5.0, 2.0, 6.0, -8.0],
            [1.0, -5.0, 1.0, 8.0],
            [7.0, 7.0, -6.0, -7.0],
            [1.0, -3.0, 7.0, 4.0],
        ]);
        let b: Matrix<4, 4> = a.inverse();
        assert_relative_eq!(a.determinant(), 532.0);
        assert_relative_eq!(a.cofactor(2, 3), -160.0);
        assert_relative_eq!(b[3][2], -160.0 / 532.0);
        assert_relative_eq!(a.cofactor(3, 2), 105.0);
        assert_relative_eq!(b[2][3], 105.0 / 532.0);
        assert_relative_eq!(
            b,
            Matrix::new([
                [0.21805, 0.45113, 0.24060, -0.04511],
                [-0.80827, -1.45677, -0.44361, 0.52068],
                [-0.07895, -0.22368, -0.05263, 0.19737],
                [-0.52256, -0.81391, -0.30075, 0.30639],
            ]),
            epsilon = 0.00001
        );

        let a: Matrix<4, 4> = Matrix::new([
            [8.0, -5.0, 9.0, 2.0],
            [7.0, 5.0, 6.0, 1.0],
            [-6.0, 0.0, 9.0, 6.0],
            [-3.0, 0.0, -9.0, -4.0],
        ]);
        assert_relative_eq!(
            a.inverse(),
            Matrix::new([
                [-0.15385, -0.15385, -0.28205, -0.53846],
                [-0.07692, 0.12308, 0.02564, 0.03077],
                [0.35897, 0.35897, 0.43590, 0.92308],
                [-0.69231, -0.69231, -0.76923, -1.92308],
            ]),
            epsilon = 0.00001
        );

        let a: Matrix<4, 4> = Matrix::new([
            [9.0, 3.0, 0.0, 9.0],
            [-5.0, -2.0, -6.0, -3.0],
            [-4.0, 9.0, 6.0, 4.0],
            [-7.0, 6.0, 6.0, 2.0],
        ]);
        assert_relative_eq!(
            a.inverse(),
            Matrix::new([
                [-0.04074, -0.07778, 0.14444, -0.22222],
                [-0.07778, 0.03333, 0.36667, -0.33333],
                [-0.02901, -0.14630, -0.10926, 0.12963],
                [0.17778, 0.06667, -0.26667, 0.33333],
            ]),
            epsilon = 0.00001
        );
    }

    #[test]
    fn matrix_mult_inverse() {
        let a: Matrix<4, 4> = Matrix::new([
            [3.0, -9.0, 7.0, 3.0],
            [3.0, -8.0, 2.0, -9.0],
            [-4.0, 4.0, 4.0, 1.0],
            [-6.0, 5.0, -1.0, 1.0],
        ]);
        let b: Matrix<4, 4> = Matrix::new([
            [8.0, 2.0, 2.0, 2.0],
            [3.0, -1.0, 7.0, 0.0],
            [7.0, 0.0, 5.0, 4.0],
            [6.0, -2.0, 0.0, 5.0],
        ]);
        let c = a * b;
        assert_relative_eq!(c * b.inverse(), a, epsilon = 0.00001);
    }
}
