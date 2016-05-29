use tools::dual_number::DualNumber;

pub struct FeedForward<'a> {
    weights: Vec<f64>,
    len_row: usize,
    bias: Vec<f64>,
    func: &'a Box<Fn(DualNumber) -> DualNumber>,
}

impl<'a> FeedForward<'a> {
    pub fn new(weights: Vec<f64>, len_row: usize, bias: Vec<f64>, func: &'a Box<Fn(DualNumber) -> DualNumber>) -> FeedForward<'a> {
        FeedForward {
            weights: weights,
            len_row: len_row,
            bias: bias,
            func: func,
        }
    }

    /// 層の出力を算出する。
    pub fn outputs(&self, inputs: &'a Vec<DualNumber>) -> Vec<DualNumber> {
        self.calc_inputs(inputs).iter().map(|&elem| {
            (self.func)(elem)
        }).collect::<Vec<DualNumber>>()
    }

    /// 重み付き総入力を算出する。
    fn calc_inputs(&self, inputs: &'a Vec<DualNumber>) -> Vec<DualNumber> {
        self.weights.chunks(inputs.len()).map(|line_weights| {
            line_weights.iter().zip(inputs.iter()).fold(DualNumber::new(0.0), |acc: DualNumber, (&weight, &input)| {
                acc + DualNumber::new(weight) * input
            })
        }).zip(self.bias.iter()).map(|(elem, &b)| elem + DualNumber::new(b)).collect::<Vec<DualNumber>>()
    }

    /// 出力層の誤差を算出する。
    pub fn calc_error_output<'b>(&self, outputs: &'b Vec<DualNumber>, teachers: &'b Vec<DualNumber>, func_error: &'b Box<Fn(&DualNumber, &DualNumber) -> DualNumber>) -> Vec<f64> {
        let mut selected = outputs.clone();
        for elem in &mut selected {
            elem.dx = 1.0;
        }

        let vec_error = teachers.iter().zip(selected.iter()).map(|(t, o)| {
            func_error(&o, &t).dx
        }).collect::<Vec<f64>>();

        let vec_output = outputs.iter().map(|&elem| {
            elem.dx
        }).collect::<Vec<f64>>();

        FeedForward::calc_hadamard(vec_error, vec_output)
    }

    /// 隠れ層の誤差を算出する。
    pub fn calc_error_hide<'b>(&mut self, outputs: &'b Vec<DualNumber>, prev_layer: &'b FeedForward, prev_errors: &'b Vec<f64>) -> Vec<f64> {
        let transposed = prev_layer.transposed_weights();

        let vec_error = transposed.chunks(prev_errors.len()).map(|line_weights| {
            line_weights.iter().zip(prev_errors.iter()).fold(0.0, |acc: f64, (&weight, &error)| {
                acc + weight * error
            })
        }).collect::<Vec<f64>>();

        let vec_output = outputs.iter().map(|&elem| {
            elem.dx
        }).collect::<Vec<f64>>();

        FeedForward::calc_hadamard(vec_error, vec_output)
    }

    /// 重み行列を成型して表示する
    pub fn print_weights(&self) {
        for line_weights in self.weights.chunks(self.len_row) {
            for weight in line_weights {
                print!(" {:.*}", 5, weight);
            }
            println!("");
        }
    }

    /// 重み行列を転置する。
    ///
    /// # Examples
    ///
    /// ```
    /// let inputs = vec![1.0,1.0];
    /// let mut layer = FeedForward::new(vec![1.0,2.0,3.0,4.0], 2, 0.0, Box::new(move |x: f64| x));
    ///
    /// let transposed = layer.transposed_weight(inputs.len());
    /// assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn transposed_weights(&self) -> Vec<f64> {
        let num_row = self.weights.len() / self.len_row;
        let mut transposed = self.weights.clone();

        for row in 0..num_row {
            for column in row..self.len_row {
                transposed.swap(row * self.len_row + column, column * num_row + row)
            }
        }

        transposed
    }

    pub fn update_weights<'b>(&mut self, errors: &'b Vec<f64>, rate: f64) {
        for i in 0..self.weights.len() {
            self.weights[i] -= rate * errors[i/self.len_row]
        }
    }

    pub fn update_bias<'b>(&mut self, errors: &'b Vec<f64>, rate: f64) {
        for i in 0..self.bias.len() {
            self.bias[i] -= rate * errors[i]
        }
    }

    fn calc_hadamard(vec1: Vec<f64>, vec2: Vec<f64>) -> Vec<f64> {
        vec1.iter().zip(vec2.iter()).map(|(&elem1, &elem2)| {
            elem1 * elem2
        }).collect::<Vec<f64>>()
    }
}
