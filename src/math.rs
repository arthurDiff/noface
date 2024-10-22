pub struct Math;

impl Math {
    pub fn mean<const C: usize, const R: usize>(set: [[f32; C]; R]) -> [f32; C] {
        set.iter().fold([0.; C], |accu, row| {
            accu.iter()
                .enumerate()
                .map(|(idx, v)| v + row[idx] / R as f32)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap()
        })
    }

    pub fn centroid_matrix<const C: usize, const R: usize>(set: [[f32; C]; R]) -> [[f32; C]; R] {
        let mean = Self::mean(set);
        set.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(col_idx, v)| v - mean[col_idx])
                    .collect::<Vec<f32>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[f32; C]>>()
            .try_into()
            .unwrap()
    }

    pub fn variance<const C: usize, const R: usize>(set: [[f32; C]; R]) -> [f32; C] {
        let mean = Self::mean(set);
        set.iter().fold([0.; C], |accu, row| {
            accu.iter()
                .enumerate()
                .map(|(idx, v)| v + (row[idx] - mean[idx]).abs().powi(2) / R as f32)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap()
        })
    }

    pub fn covariance<const C: usize, const R: usize>(
        set_a: [[f32; C]; R],
        set_b: [[f32; C]; R],
    ) -> f32 {
        let (a_mean, b_mean) = (Self::mean(set_a), Self::mean(set_b));
        set_a
            .iter()
            .zip(set_b.iter())
            .fold(0., |accu, (a_row, b_row)| {
                accu + a_row
                    .iter()
                    .zip(b_row.iter())
                    .enumerate()
                    .fold(0., |accu, (idx, (a_v, b_v))| {
                        accu + (a_v - a_mean[idx]) * (b_v - b_mean[idx])
                    })
            })
            / R as f32
    }

    pub fn covariance_matrix<const C: usize, const R: usize>(
        set_a: [[f32; C]; R],
        set_b: [[f32; C]; R],
    ) -> [[f32; C]; C] {
        let (c_a, c_b) = (Math::centroid_matrix(set_a), Math::centroid_matrix(set_b));
        (0..C)
            .map(|c_idx| {
                let a_col = c_a.map(|row| row[c_idx]);
                a_col
                    .iter()
                    .enumerate()
                    .fold([0.; C], |accu, (r_idx, row_val)| {
                        accu.iter()
                            .enumerate()
                            .map(|(cov_c_idx, v)| v + (row_val * c_b[r_idx][cov_c_idx]) / R as f32)
                            .collect::<Vec<f32>>()
                            .try_into()
                            .unwrap()
                    })
            })
            .collect::<Vec<[f32; C]>>()
            .try_into()
            .unwrap()
    }

    pub fn tridiagonalization<const N: usize>(set: [[f32; N]; N]) -> [[f32; N]; N] {
        (0..N)
            .fold(set, |tridi_accu, idx| {
                if idx == N-1 { return tridi_accu; }
                let col = tridi_accu.map(|row| row[idx]);
                let col_vec = &col[idx+1..];
                let row_norm = col_vec.iter().fold(0., |accu, v| accu + v * v).sqrt();

                let reflection_vec =
                   col_vec 
                        .iter()
                        .enumerate()
                        .map(|(cv_idx, v)| if cv_idx == 0 { v + {if *v < 0. {-1.} else {1.}} * row_norm } else { *v });
                let reflect_norm = reflection_vec
                    .clone()
                    .fold(0., |rn_accu, v| rn_accu + v * v)
                    .sqrt();

                let normalized_vec = reflection_vec.map(|v| v / reflect_norm).chain([0.]);

                let householder_matrix =
                    normalized_vec.clone().enumerate().map(|(outer_idx, v)| {
                        normalized_vec
                            .clone()
                            .enumerate()
                            .map(move |(inner_idx, inner_v)| {if outer_idx == inner_idx{1.}else{0.}} - 2.* v * inner_v).collect::<Vec<f32>>()
                    }).collect::<Vec<Vec<f32>>>();

                tridi_accu.iter().enumerate().map(|(outer_r_idx, row)|{
                        if outer_r_idx < idx {return *row;} 
                
                        row.iter().enumerate().map(|(c_idx, v)|{
                            if c_idx < idx {return *v;}
          
                            let mul_val = householder_matrix[outer_r_idx - idx].iter().enumerate().fold(0., |hm_accu, (hm_idx, hm_v)|{
                               hm_accu + *hm_v * tridi_accu[outer_r_idx + hm_idx][c_idx]
                            });
                            mul_val
                        }).collect::<Vec<f32>>().try_into().unwrap()
                    }).collect::<Vec<[f32;N]>>().try_into().unwrap()
                    // need transposed householder_matrix dot
            })
    }

    // tridiagonalization + divide and conquer methods
    pub fn eigenvalues<const N: usize>(set: [[f32; N]; N]) -> [f32; N] {
        todo!()
    }

    // Singular Value Decomposition
    pub fn svd<const C: usize>(set: [[f32; C]; C]) {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::Math;

    #[test]
    fn get_correct_covariance_matrix() {
        let a = [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]];
        let b = [[11., 12.], [13., 14.], [15., 16.], [17., 18.], [19., 20.]];

        let cov_mat = Math::covariance_matrix(a, b);

        for row in cov_mat {
            assert_eq!(row[0], 8.);
            assert_eq!(row[1], 8.);
        }
    }
}
