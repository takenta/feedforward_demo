extern crate rand;

use std::io;
use rand::Rng;

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
    let weights_hide: Vec<f64> = gen_rand_weights(num_input*num_node_hide, 0.0, 0.1);
    let weights_output: Vec<f64> = gen_rand_weights(num_input*num_node_output, 0.0, 0.1);

    // 活性化関数
    let func_hide: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x.max(0.0)); // 標準正規化関数
    let func_output: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x); // 恒等写像

    // レイヤーの生成
    let mut layer_hide: FeedForward = FeedForward::new(weights_hide, 0.0, &func_hide); // 隠れ層
    let layer_output: FeedForward = FeedForward::new(weights_output, 0.0, &func_output); // 出力層

    // 誤差関数（二乗誤差；回帰・入力が連続したデータである時に有効）
    let diff_func: Box<Fn(&Vec<f64>, &Vec<f64>) -> f64> = Box::new(move |outputs: &Vec<f64>, teacheres: &Vec<f64>| {
        outputs.iter().zip(teacheres.iter()).fold(0.01, |acc: f64, (&output, &teacher)| {
            acc + (output - teacher).abs().powf(2.0)
        }) * 0.5
    });

    // 標準入力用の文字列
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
        let outputs: Vec<f64> = layer_output.output(&layer_hide.output(&inputs));
        for output in &outputs {
            print!(" {:.*}", 5, (output * 1000.0).round() * 0.001);
        };
        println!("\n");

        println!("answer:");
        for answer in teacher {
            print!(" {:.*}", 5, (answer * 1000.0).round() * 0.001);
        }
        println!("\n");

        println!("diff:\n {:.*}", 5, (calc_diff(&outputs, &teacher, &diff_func) * 1000.0).round() * 0.001);
        println!("");

        println!("Please input modification for hide layer (ex: 0 0 0.01):");

        let mut input_string: String = String::new();
        io::stdin().read_line(&mut input_string).expect("Failed to read line");

        if input_string.as_str() == "0" { break 'main; }

        let modifications: Vec<&str> = input_string.split_whitespace().collect();
        let target_row: usize = modifications[0].trim().parse().expect("Please input a number!");
        let target_column: usize = modifications[1].trim().parse().expect("Please input a number!");
        let new_weight: f64 = modifications[2].trim().parse().expect("Please input a number!");

        layer_hide.update_weights(target_row, target_column, inputs.len(), new_weight);
    }
}

// ========================================================================= //

pub fn calc_hadamard(vec1: Vec<f64>, vec2: Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).fold(0 as f64, |acc: f64, (&elem1, &elem2)| {
        acc + elem1 * elem2
    })
}

pub fn gen_rand_weights(num: usize, low: f64, high: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    rng.gen_iter::<f64>().take(num).map(|item| (high - low)*item + low).collect::<Vec<f64>>()
}

pub fn calc_diff<'a>(output: &'a Vec<f64>, teacher: &'a Vec<f64>, func: &'a Box<Fn(&Vec<f64>, &Vec<f64>) -> f64>) -> f64 {
    func(output, teacher)
}

struct FeedForward<'a> {
    weights: Vec<f64>,
    bias: f64,
    func: &'a Box<Fn(f64) -> f64>,
}

impl<'a> FeedForward<'a> {
    fn new(weights: Vec<f64>, bias: f64, func: &'a Box<Fn(f64) -> f64>) -> FeedForward<'a> {
        FeedForward {
            weights: weights,
            bias: bias,
            func: func,
        }
    }

    fn output(&self, inputs: &'a Vec<f64>) -> Vec<f64> {
        if (self.weights.len() % inputs.len()) != 0 {
            panic!("length of arguments are mismatched.");
        };

        self.weights.chunks(inputs.len()).map(|line_weights| {
            (self.func)(line_weights.iter().zip(inputs.iter()).fold(0.0, |acc: f64, (&weight, &input)| {
                acc + weight * input
            }) + self.bias)
        }).collect::<Vec<f64>>()
    }

    fn print_weights(&'a self, length: &'a usize) {
        for line_weights in self.weights.chunks(*length) {
            for weight in line_weights {
                print!(" {:.*}", 2, weight);
            }
            println!("");
        }
    }

    fn update_weights(&mut self, row: usize, column: usize, num_column: usize, new_weight: f64) {
        self.weights = self.weights.iter().enumerate().map(|(index, elem)| {
            match row * num_column + column {
                x if x == index => new_weight,
                _ => *elem,
            }
        }).collect::<Vec<f64>>();
    }
}
