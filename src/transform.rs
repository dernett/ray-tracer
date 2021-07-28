use crate::matrix::Matrix;

impl Matrix<4, 4> {
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Matrix::new([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn scaling(x: f32, y: f32, z: f32) -> Self {
        Matrix::new([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotation_x(r: f32) -> Self {
        Matrix::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, r.cos(), -r.sin(), 0.0],
            [0.0, r.sin(), r.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotation_y(r: f32) -> Self {
        Matrix::new([
            [r.cos(), 0.0, r.sin(), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-r.sin(), 0.0, r.cos(), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotation_z(r: f32) -> Self {
        Matrix::new([
            [r.cos(), -r.sin(), 0.0, 0.0],
            [r.sin(), r.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn shearing(
        xy: f32,
        xz: f32,
        yx: f32,
        yz: f32,
        zx: f32,
        zy: f32,
    ) -> Self {
        Matrix::new([
            [1.0, xy, xz, 0.0],
            [yx, 1.0, yz, 0.0],
            [zx, zy, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

#[cfg(test)]
mod tests_transform {
    use super::*;
    use crate::tuple::Tuple;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    #[test]
    fn translation_matrix() {
        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let p = Tuple::point(-3.0, 4.0, 5.0);
        assert_relative_eq!(transform * p, Tuple::point(2.0, 1.0, 7.0));

        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let inv = transform.inverse();
        let p = Tuple::point(-3.0, 4.0, 5.0);
        assert_relative_eq!(inv * p, Tuple::point(-8.0, 7.0, 3.0));

        let transform = Matrix::translation(5.0, -3.0, 2.0);
        let v = Tuple::vector(-3.0, 4.0, 5.0);
        assert_relative_eq!(transform * v, v);
    }

    #[test]
    fn scaling_matrix() {
        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let p = Tuple::point(-4.0, 6.0, 8.0);
        assert_relative_eq!(transform * p, Tuple::point(-8.0, 18.0, 32.0));

        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let v = Tuple::vector(-4.0, 6.0, 8.0);
        assert_relative_eq!(transform * v, Tuple::vector(-8.0, 18.0, 32.0));

        let transform = Matrix::scaling(2.0, 3.0, 4.0);
        let inv = transform.inverse();
        let v = Tuple::vector(-4.0, 6.0, 8.0);
        assert_relative_eq!(inv * v, Tuple::vector(-2.0, 2.0, 2.0));
    }

    #[test]
    fn reflection_eq_scaling() {
        let transform = Matrix::scaling(-1.0, 1.0, 1.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(-2.0, 3.0, 4.0));
    }

    #[test]
    fn rotating_point_x() {
        let p = Tuple::point(0.0, 1.0, 0.0);
        let half_quarter = Matrix::rotation_x(PI / 4.0);
        let full_quarter = Matrix::rotation_x(PI / 2.0);
        assert_relative_eq!(
            half_quarter * p,
            Tuple::point(0.0, f32::sqrt(2.0) / 2.0, f32::sqrt(2.0) / 2.0),
        );
        assert_relative_eq!(full_quarter * p, Tuple::point(0.0, 0.0, 1.0));

        let p = Tuple::point(0.0, 1.0, 0.0);
        let half_quarter = Matrix::rotation_x(PI / 4.0);
        let inv = half_quarter.inverse();
        assert_relative_eq!(
            inv * p,
            Tuple::point(0.0, f32::sqrt(2.0) / 2.0, -f32::sqrt(2.0) / 2.0),
        );
    }

    #[test]
    fn rotation_point_y() {
        let p = Tuple::point(0.0, 0.0, 1.0);
        let half_quarter = Matrix::rotation_y(PI / 4.0);
        let full_quarter = Matrix::rotation_y(PI / 2.0);
        assert_relative_eq!(
            half_quarter * p,
            Tuple::point(f32::sqrt(2.0) / 2.0, 0.0, f32::sqrt(2.0) / 2.0),
        );
        assert_relative_eq!(full_quarter * p, Tuple::point(1.0, 0.0, 0.0));
    }

    #[test]
    fn rotation_point_z() {
        let p = Tuple::point(0.0, 1.0, 0.0);
        let half_quarter = Matrix::rotation_z(PI / 4.0);
        let full_quarter = Matrix::rotation_z(PI / 2.0);
        assert_relative_eq!(
            half_quarter * p,
            Tuple::point(-f32::sqrt(2.0) / 2.0, f32::sqrt(2.0) / 2.0, 0.0),
        );
        assert_relative_eq!(full_quarter * p, Tuple::point(-1.0, 0.0, 0.0));
    }

    #[test]
    fn shearing_matrix() {
        let transform = Matrix::shearing(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(5.0, 3.0, 4.0));

        let transform = Matrix::shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(6.0, 3.0, 4.0));

        let transform = Matrix::shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(2.0, 5.0, 4.0));

        let transform = Matrix::shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(2.0, 7.0, 4.0));

        let transform = Matrix::shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(2.0, 3.0, 6.0));

        let transform = Matrix::shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = Tuple::point(2.0, 3.0, 4.0);
        assert_relative_eq!(transform * p, Tuple::point(2.0, 3.0, 7.0));
    }

    #[test]
    fn transformations_sequence() {
        let p = Tuple::point(1.0, 0.0, 1.0);
        let a = Matrix::rotation_x(PI / 2.0);
        let b = Matrix::scaling(5.0, 5.0, 5.0);
        let c = Matrix::translation(10.0, 5.0, 7.0);
        let p2 = a * p;
        let p3 = b * p2;
        let p4 = c * p3;
        let t = c * b * a;
        assert_relative_eq!(
            p2,
            Tuple::point(1.0, -1.0, 0.0),
            epsilon = 0.00001
        );
        assert_relative_eq!(
            p3,
            Tuple::point(5.0, -5.0, 0.0),
            epsilon = 0.00001
        );
        assert_relative_eq!(
            p4,
            Tuple::point(15.0, 0.0, 7.0),
            epsilon = 0.00001
        );
        assert_relative_eq!(
            t * p,
            Tuple::point(15.0, 0.0, 7.0),
            epsilon = 0.00001
        );
    }
}
