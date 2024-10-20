use nalgebra::Matrix3;

use crate::math::Math;

const KEY_POINTS_LEN: usize = 5;
const ARC_FACE_DST: KeyPoints = KeyPoints([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]);

#[derive(Debug, Clone)]
pub struct KeyPoints(pub [[f32; 2]; KEY_POINTS_LEN]);

impl KeyPoints {
    fn mean(&self) -> [f32; 2] {
        Math::mean(self.0)
    }

    #[allow(dead_code)]
    fn covariance(&self, other: &Self) -> f32 {
        Math::covariance(self.0, other.0)
    }

    fn covariance_matrix(&self, other: &Self) -> [[f32; 2]; 2] {
        Math::covariance_matrix(self.0, other.0)
    }

    pub fn umeyama(&self, dst: &Self) -> nalgebra::Matrix3<f32> {
        use nalgebra::{ArrayStorage, Matrix, Matrix1x2, Matrix2, Matrix2x1};
        use std::ops::Mul;
        let [src_x_mean, src_y_mean] = self.mean();
        let [dst_x_mean, dst_y_mean] = dst.mean();
        let (src_dmean, dst_dmean) = (
            Matrix::from_array_storage(ArrayStorage(
                self.0.map(|[x, y]| [x - src_x_mean, y - src_y_mean]),
            )),
            Matrix::from_array_storage(ArrayStorage(
                dst.0.map(|[x, y]| [x - dst_x_mean, y - dst_y_mean]),
            )),
        );
        let a = std::ops::Mul::mul(dst_dmean, &src_dmean.transpose()) / 5.;
        let svd = Matrix::svd(a, true, true);
        let determinant = a.determinant();

        let mut d = [1f32; 2];
        if determinant < 0. {
            d[1] = -1.;
        }

        let mut t = Matrix2::<f32>::identity();
        let (s, u, v) = (svd.singular_values, svd.u.unwrap(), svd.v_t.unwrap());

        let rank = a.rank(0.00001f32);
        if rank == 0 {
            panic!("Matrix rank is 0");
        }

        if rank == 1 {
            if u.determinant() * v.determinant() > 0. {
                u.mul_to(&v, &mut t);
            } else {
                let s = d[1];
                d[1] = -1.;
                let dg = Matrix2::<f32>::new(d[0], 0., 0., d[1]);

                let udg = u.mul(&dg);
                udg.mul_to(&v, &mut t);
                d[1] = s;
            }
        } else {
            let dg = Matrix2::<f32>::new(d[0], 0., 0., d[1]);
            let udg = u.mul(&dg);
            udg.mul_to(&v, &mut t);
        }

        let ddd = Matrix1x2::new(d[0], d[1]);
        let d_x_s = ddd.mul(s);

        let (var0, var1) = (
            src_dmean.remove_row(0).variance(),
            src_dmean.remove_row(1).variance(),
        );

        let var_sum = var0 + var1;

        let scale = d_x_s.get((0, 0)).unwrap() / var_sum;

        let (dst_mean, src_mean) = (
            Matrix2x1::<f32>::new(dst_x_mean, dst_y_mean),
            Matrix2x1::<f32>::new(src_x_mean, src_y_mean),
        );
        let t_x_src_mean = t.mul(&src_mean);

        let xxx = scale * t_x_src_mean;
        let yyy = dst_mean - xxx;

        let (m13, m23) = (*yyy.get(0).unwrap(), *yyy.get(1).unwrap());

        let m00x22 = t * scale;

        let (m11, m21, m12, m22) = (m00x22.m11, m00x22.m21, m00x22.m12, m00x22.m22);

        Matrix3::<f32>::new(m11, m12, m13, m21, m22, m23, 0., 0., 1.)
    }

    /// (f32, f32, f32) : (R:Rotation Matrix, c:Scale Factor, t: Translation Vector)
    pub fn umeyama_v2(&self, dst: &Self) -> (f32, f32, f32) {
        let _cov_mat = self.covariance_matrix(dst);

        //Singular Value Decomposition
        todo!();
    }

    pub fn umeyama_to_arc(&self) -> Matrix3<f32> {
        self.umeyama(&ARC_FACE_DST)
    }
}

impl std::ops::Deref for KeyPoints {
    type Target = [[f32; 2]; KEY_POINTS_LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for KeyPoints {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod test {
    use super::KeyPoints;

    #[test]
    fn get_correct_covariance_matrix() {
        let test_kp = KeyPoints([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]]);
        let test_kp_2 = KeyPoints([[11., 12.], [13., 14.], [15., 16.], [17., 18.], [19., 20.]]);

        let cov_mat = test_kp.covariance_matrix(&test_kp_2);

        for row in cov_mat {
            assert_eq!(row[0], 10.);
            assert_eq!(row[1], 10.);
        }
    }
}
