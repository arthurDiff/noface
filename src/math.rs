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

    pub fn covariance_matrix<const C: usize, const R: usize, const M: usize>(
        set_a: [[f32; C]; R],
        set_b: [[f32; C]; R],
    ) -> [[f32; M]; M] {
        let (a_mean, b_mean) = (Self::mean(set_a), Self::mean(set_b));
        let deviations: [[f32; M]; R] = set_a
            .iter()
            .zip(set_b.iter())
            .map(|(a_r, b_r)| {
                let dev_row: [f32; M] = a_r
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| v - a_mean[idx])
                    .chain(b_r.iter().enumerate().map(|(idx, v)| v - b_mean[idx]))
                    .collect::<Vec<f32>>()
                    .try_into()
                    .unwrap();
                dev_row
            })
            .collect::<Vec<[f32; M]>>()
            .try_into()
            .unwrap();
        (0..M)
            .map(|col| {
                let dev_col = deviations.map(|mat| mat[col]);
                dev_col
                    .iter()
                    .enumerate()
                    .fold([0.; M], |accu, (col_idx, col_val)| {
                        let m_row = accu
                            .iter()
                            .enumerate()
                            .map(|(idx, v)| v + col_val * deviations[col_idx][idx] / M as f32)
                            .collect::<Vec<f32>>()
                            .try_into()
                            .unwrap();
                        m_row
                    })
            })
            .collect::<Vec<[f32; M]>>()
            .try_into()
            .unwrap()
    }
}
