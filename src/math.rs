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

    // Need Validation - Source Material Contradicted Itself
    pub fn tridiagonalization<const N: usize>(set: [[f32; N]; N]) -> [[f32; N]; N] {
        (0..N)
            .fold(set, |tridiag_accu, idx| {
                if idx == N-1 { return tridiag_accu; }
                let col_vec = &tridiag_accu.map(|row| row[idx])[idx+1..];
                let col_norm = col_vec.iter().fold(0., |accu, v| accu + v * v).sqrt();
            
                let mut reflection_vec = col_vec.to_vec();
                reflection_vec[0] += {if reflection_vec[0] > 0. {-1.} else { 1. }} * col_norm;

                let reflection_norm = reflection_vec
                    .clone().iter().fold(0., |rn_accu, v| rn_accu + v * v)
                    .sqrt();

                let mut normalized_vec = if reflection_norm != 0.{ reflection_vec.iter().map(|v| v / reflection_norm).collect::<Vec<f32>>() }else{ reflection_vec };
                normalized_vec.insert(0, 0.);

                let householder_matrix =
                    normalized_vec.clone().iter().enumerate().map(|(r_idx, r_v)| {
                        normalized_vec
                            .clone().iter()
                            .enumerate()
                            .map(move |(c_idx, c_v)| {if r_idx == c_idx{1.}else{0.}} - 2.* r_v * c_v).collect::<Vec<f32>>()
                    }).collect::<Vec<Vec<f32>>>();
                
                tridiag_accu.iter().enumerate().map(|(outer_r_idx, row)|{
                        if outer_r_idx < idx {return *row;} 
                
                        row.iter().enumerate().map(|(c_idx, v)|{
                            if c_idx < idx {return *v;}
          
                            householder_matrix[outer_r_idx - idx].iter().enumerate().fold(0., |hm_accu, (hm_idx, hm_v)|{
                               hm_accu + *hm_v * tridiag_accu[idx + hm_idx][c_idx] * householder_matrix[hm_idx][c_idx - idx]
                            })
        
                        }).collect::<Vec<f32>>().try_into().unwrap()
                    }).collect::<Vec<[f32;N]>>().try_into().unwrap()
            })
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
