fn main() {
    // 教師データ
    let inputs: Vec<f64> = vec![1.0,2.0,3.0,4.0,5.0]; // 入力
    let teacher: Vec<f64> = vec![1.0,2.0,3.0,4.0,5.0]; // 期待される出力

    // ネットワークの生成
    // 重み
    let mut weights_hide: Vec<f64> = vec![
        0.1, 0.0, 0.2, 0.0, 0.1,
        0.1, 0.4, 0.1, 0.2, 0.1,
        0.8, 0.1, 0.4, 0.0, 0.1,
        0.1, 0.2, 0.4, 0.2, 0.2,
        0.1, 0.6, 0.1, 0.2, 0.4
    ];
    let mut weights_output: Vec<f64> = vec![
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
    ];

    // 活性化関数
    let func_hide: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x.max(0.0));
    let func_output: Box<Fn(f64) -> f64> = Box::new(move |x: f64| x);

    // レイヤーの生成
    let layer_hide: Layer = Layer::new(&mut weights_hide, 0.0, &func_hide); // 標準正規化関数
    let layer_output: Layer = Layer::new(&mut weights_output, 0.0, &func_output); // 恒等写像

    // レイヤー２枚で演算
    print!("result: ( ");
    let outputs: Vec<f64> = layer_output.output(&layer_hide.output(&inputs));
    for output in &outputs {
        print!("{} ", output);
    };
    println!(")");

    // 二乗誤差（回帰・入力が連続したデータである時に有効）
    let diff_func: Box<Fn(&Vec<f64>, &Vec<f64>) -> f64> = Box::new(move |outputs: &Vec<f64>, teacheres: &Vec<f64>| {
        outputs.iter().zip(teacheres.iter()).fold(0.01, |acc: f64, (&output, &teacher)| {
            acc + (output - teacher).abs().powf(2.0)
        }) * 0.5
    });
    println!("diff: {}", Layer::calc_diff(&outputs, &teacher, &diff_func));
}

struct Layer<'a> {
    weights: &'a Vec<f64>,
    bias: f64,
    func: &'a Box<Fn(f64) -> f64>,
}

impl<'a> Layer<'a> {
    fn new(weights: &'a mut Vec<f64>, bias: f64, func: &'a Box<Fn(f64) -> f64>) -> Layer<'a> {
        Layer {
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

    fn calc_diff(output: &'a Vec<f64>, teacher: &'a Vec<f64>, func: &'a Box<Fn(&Vec<f64>, &Vec<f64>) -> f64>) -> f64 {
        func(output, teacher)
    }
}
