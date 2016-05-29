extern crate rand;
mod tools;

use std::thread;
use std::time::Duration;
use rand::Rng;
use tools::dual_number::DualNumber;

fn main() {
    // 教師データ
    let mut inputs: Vec<DualNumber> = vec![DualNumber::new(1.0),DualNumber::new(2.0),DualNumber::new(3.0),DualNumber::new(4.0),DualNumber::new(5.0),DualNumber::new(6.0),DualNumber::new(7.0),DualNumber::new(8.0),DualNumber::new(9.0)]; // 入力
    // let mut inputs: Vec<DualNumber> = vec![DualNumber::new(1.0),DualNumber::new(2.0),DualNumber::new(3.0),DualNumber::new(4.0),DualNumber::new(5.0)]; // 入力
    // let mut inputs: Vec<DualNumber> = vec![DualNumber::new(5.0),DualNumber::new(4.0),DualNumber::new(3.0),DualNumber::new(2.0),DualNumber::new(1.0)];
    let teachers: Vec<DualNumber> = inputs.clone(); // 期待される出力

    // 入力の次元数と層のノード数
    let num_input: usize = inputs.len();
    let num_node_hide: usize = 10;
    let num_node_output: usize = num_input;

    // ネットワークの生成
    // 重み
    let weights_hide_1: Vec<f64> = gen_rand_weights(num_input*num_node_hide, 0.0, 0.10);
    let weights_hide_2: Vec<f64> = gen_rand_weights(num_node_hide*num_node_hide, 0.0, 0.10);
    let weights_output: Vec<f64> = gen_rand_weights(num_node_hide*num_node_output, 0.0, 0.10);

    // 活性化関数
    // let func_hide_sig: Box<Fn(DualNumber) -> DualNumber> = Box::new(move |x: DualNumber| x.max(DualNumber::new(0.0))); // 標準正規化関数
    let func_hide_rel: Box<Fn(DualNumber) -> DualNumber> = Box::new(move |x: DualNumber| (DualNumber::new(1.0) - (-x).exp()).recip()); // 標準シグモイド関数
    let func_output: Box<Fn(DualNumber) -> DualNumber> = Box::new(move |x: DualNumber| x); // 恒等写像

    // レイヤーの生成
    let mut layer_hide_1: FeedForward = FeedForward::new(weights_hide_1, num_input, gen_rand_weights(num_node_hide, 1.0, 1.0), &func_hide_rel); // 隠れ層
    let mut layer_hide_2: FeedForward = FeedForward::new(weights_hide_2, num_node_hide, gen_rand_weights(num_node_hide, 1.0, 1.0), &func_hide_rel); // 隠れ層
    let mut layer_output: FeedForward = FeedForward::new(weights_output, num_node_hide, gen_rand_weights(num_node_output, 1.0, 1.0), &func_output); // 出力層

    // 誤差関数（二乗誤差；回帰・入力が連続したデータである時に有効）
    let error_func: Box<Fn(&DualNumber, &DualNumber) -> DualNumber> = Box::new(move |&output, &teacher| (output-teacher).powf(2.0) * DualNumber::new(0.5));

    let mut count = 0;

    'main: loop {
        println!("");

        count += 1;
        println!("====== {:4} ======", &count);

        // 入力層を出力
        println!("input:");
        for elem in &inputs {
            print!(" {:.*}", 5, elem.x);
        }
        println!("\n");

        // 隠れ層を出力
        println!("weights hide:");
        layer_hide_1.print_weights();
        println!("");

        // 隠れ層を出力
        println!("weights hide:");
        layer_hide_2.print_weights();
        println!("");

        // 出力層を出力
        println!("weights output:");
        layer_output.print_weights();
        println!("");

        for input in &mut inputs {
            input.select()
        }

        // レイヤー２枚で演算
        println!("result:");
        let outputs_hide_1: Vec<DualNumber> = layer_hide_1.outputs(&inputs);
        let outputs_hide_2: Vec<DualNumber> = layer_hide_2.outputs(&outputs_hide_1);
        let outputs: Vec<DualNumber> = layer_output.outputs(&outputs_hide_2);
        for elem in &outputs {
            print!(" {:.*}", 5, elem.x);
        }
        println!("\n");

        // 教師データの表示
        println!("teachers:");
        for elem in &teachers {
            print!(" {:.*}", 5, elem.x);
        }
        println!("\n");

        // 誤差の算出
        let errors_output = layer_output.calc_error_output(&outputs, &teachers, &error_func);
        let errors_hide_2 = layer_hide_2.calc_error_hide(&outputs_hide_2, &layer_output, &errors_output);
        let errors_hide_1 = layer_hide_1.calc_error_hide(&outputs_hide_1, &layer_hide_2, &errors_hide_2);

        // 出力層の誤差の表示
        let loss = outputs.iter().zip(teachers.iter()).fold(DualNumber::new(0.0), |acc, (o, t)| {
            acc + error_func(o, t)
        });
        println!("loss:\n {:?}", &loss.x);
        println!("");

        if loss.x < 0.001 { break 'main; }

        // 重みの更新
        let rate: f64 = 0.15;
        layer_output.update_weights(&errors_output, rate);
        layer_output.update_bias(&errors_output, rate);
        layer_hide_2.update_weights(&errors_hide_2, rate);
        layer_hide_2.update_bias(&errors_hide_2, rate);
        layer_hide_1.update_weights(&errors_hide_1, rate);
        layer_hide_1.update_bias(&errors_hide_1, rate);

        thread::sleep(Duration::from_millis(100));

        if count >= 10000 { break 'main; }
    }
}


// ========================================================================= //


pub fn calc_hadamard(vec1: Vec<f64>, vec2: Vec<f64>) -> Vec<f64> {
    vec1.iter().zip(vec2.iter()).map(|(&elem1, &elem2)| {
        elem1 * elem2
    }).collect::<Vec<f64>>()
}

pub fn gen_rand_weights(num: usize, low: f64, high: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    rng.gen_iter::<f64>().take(num).map(|item| (high - low)*item + low).collect::<Vec<f64>>()
}

/// 行列を成型して表示する。
pub fn print_matrix<'a>(matrix: &'a Vec<f64>, length: &'a usize) {
    for line in matrix.chunks(*length) {
        for item in line {
            print!(" {:.*}", 2, item);
        }
        println!("");
    }
}

// ========================================================================= //


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
    fn calc_error_output<'b>(&self, outputs: &'b Vec<DualNumber>, teachers: &'b Vec<DualNumber>, func_error: &'b Box<Fn(&DualNumber, &DualNumber) -> DualNumber>) -> Vec<f64> {
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

        calc_hadamard(vec_error, vec_output)
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

        calc_hadamard(vec_error, vec_output)
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
}
