extern crate rand;
mod tools;

use std::io;
use rand::Rng;
use tools::dual_number::DualNumber;

fn main() {
    // 入力の次元数と層のノード数
    let num_input: usize = 5;
    let num_node_hide: usize = 5;
    let num_node_output: usize = 5;

    // 教師データ
    let inputs: Vec<f64> = gen_rand_weights(num_input, 0.0, 1.0); // 入力
    let teacher: &Vec<f64> = &inputs; // 期待される出力

    // ネットワークの生成
    // 重み
    let weights_hide: Vec<f64> = gen_rand_weights(num_input*num_node_hide, 0.0, 1.0);
    let weights_output: Vec<f64> = gen_rand_weights(num_input*num_node_output, 0.0, 1.0);

    // 活性化関数
    let func_hide: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x.max(0.0)); // 標準正規化関数
    let func_output: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x); // 恒等写像

    // レイヤーの生成
    let layer_hide: FeedForward = FeedForward::new(weights_hide, inputs.len(), 0.0, &func_hide); // 隠れ層
    let mut layer_output: FeedForward = FeedForward::new(weights_output, inputs.len(), 0.0, &func_output); // 出力層

    // 誤差関数（二乗誤差；回帰・入力が連続したデータである時に有効）
    let error_func: Box<Fn(&Vec<f64>, &Vec<f64>) -> f64> = Box::new(move |outputs: &Vec<f64>, teachers: &Vec<f64>| {
        outputs.iter().zip(teachers.iter()).fold(0.01, |acc: f64, (&output, &teacher)| {
            acc + (output - teacher).abs().powf(2.0)
        }) * 0.5
    });

    'main: loop {
        println!("");

        println!("===============");

        println!("input:");
        for input in &inputs {
            print!(" {:.*}", 5, (input * 1000.0).round() * 0.001);
        }
        println!("\n");

        println!("weights hide:");
        layer_hide.print_weights(&inputs.len());
        println!("");

        println!("weights output:");
        layer_output.print_weights(&inputs.len());
        println!("");

        // レイヤー２枚で演算
        println!("result:");
        let outputs: Vec<f64> = layer_output.outputs(&layer_hide.outputs(&inputs));
        for output in &outputs {
            print!(" {:.*}", 5, (output * 1000.0).round() * 0.001);
        };
        println!("\n");

        println!("answer:");
        for answer in teacher {
            print!(" {:.*}", 5, (answer * 1000.0).round() * 0.001);
        }
        println!("\n");

        println!("diff:\n {:.*}", 5, (calc_error(&outputs, &teacher, &error_func) * 1000.0).round() * 0.001);
        println!("");

        println!("Please input modification for output layer (ex: 0 0 0.01) (EXIT: halt):");

        let mut input_string: String = String::new();
        io::stdin().read_line(&mut input_string).expect("Failed to read line");

        if input_string.trim() == "halt" { break 'main; }

        let modifications: Vec<&str> = input_string.split_whitespace().collect();
        // let target: &str = modifications[0].trim();
        let target_row: usize = modifications[0].trim().parse().expect("Please input a number!");
        let target_column: usize = modifications[1].trim().parse().expect("Please input a number!");
        let new_weight: f64 = modifications[2].trim().parse().expect("Please input a number!");

        layer_output.update_weights(target_row, target_column, new_weight);
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

pub fn calc_error<'a>(output: &'a Vec<f64>, teacher: &'a Vec<f64>, func: &'a Box<Fn(&Vec<f64>, &Vec<f64>) -> f64>) -> f64 {
    func(output, teacher)
}

/// 行列を成型して表示する。
pub fn print_matrix<'a>(matrix: &'a Vec<f64>, length: &'a usize) {
    for line in matrix.chunks(*length) {
        for item in line {
            println!(" {:.*}", 2, item);
        }
        println!("");
    }
}

/// 関数を表すクロージャを受け取って、その関数の数値微分を表すクロージャを返す。
pub fn diff_func(func: Box<Fn(DualNumber) -> DualNumber>) -> Box<Fn(f64) -> f64> {
    Box::new(move |x: f64| func(DualNumber::new(x)).dx)
}


// ========================================================================= //


pub struct FeedForward<'a> {
    weights: Vec<f64>,
    len_row: usize,
    bias: f64,
    func: &'a Box<Fn(f64) -> f64>,
}

impl<'a> FeedForward<'a> {
    pub fn new(weights: Vec<f64>, len_row: usize, bias: f64, func: &'a Box<Fn(f64) -> f64>) -> FeedForward<'a> {
        FeedForward {
            weights: weights,
            len_row: len_row,
            bias: bias,
            func: func,
        }
    }

    /// 層の出力を算出する。
    pub fn outputs(&self, inputs: &'a Vec<f64>) -> Vec<f64> {
        self.calc_inputs(inputs).iter().map(|&elem| (self.func)(elem)).collect::<Vec<f64>>()
    }

    /// 重み付き総入力を算出する。
    fn calc_inputs(&self, inputs: &'a Vec<f64>) -> Vec<f64> {
        self.weights.chunks(inputs.len()).map(|line_weights| {
            line_weights.iter().zip(inputs.iter()).fold(0.0, |acc: f64, (&weight, &input)| {
                acc + weight * input
            }) + self.bias
        }).collect::<Vec<f64>>()
    }

    /// 重み行列を成型して表示する
    pub fn print_weights(&'a self, length: &'a usize) {
        for line_weights in self.weights.chunks(*length) {
            for weight in line_weights {
                print!(" {:.*}", 2, weight);
            }
            println!("");
        }
    }

    /// 行と列を指定して、その箇所の重みを更新する。
    pub fn update_weights(&mut self, row: usize, column: usize, new_weight: f64) {
        let target = row * self.len_row + column;
        self.weights[target] = new_weight;
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
                transposed.swap(row * self.len_row + column, column * self.len_row + row)
            }
        }

        transposed
    }

    // pub fn calc_output_error(&self, inputs: &'a Vec<f64>, teachers: &'a Vec<f64>, func_error: &'a Box<Fn(&DualNumber, &DualNumber) -> DualNumber>) -> &'a Vec<f64> {

    //     // let diff_func_error = teachers.iter().map(|&t| {
    //     //     diff_func(Box::new(move |x| func_error(&t, &x)))
    //     // }).collect::<Vec<Fn(f64) -> f64>>();
    //     let diff_func_output = diff_func(*self.func);


    //     let vec_error = teachers.iter().zip(inputs.iter()).map(|&t, &i| {
    //         let dn_teacher: DualNumber = DualNumber::new(t);
    //         let dn_input: DualNumber = DualNumber::new(i);

    //         func_error(dn_teacher, dn_input).dx
    //     }).collect::<Vec<f64>>();
    //     let vec_output = inputs.iter().map(|&i| {
    //         let dn_input: DualNumber = DualNumber::new(i);

    //         self.func()
    //     }).collect::<Vec<f64>>();
    //     &calc_hadamard(vec_error, vec_output)
    // }
}
