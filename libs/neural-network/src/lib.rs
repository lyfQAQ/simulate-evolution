use rand::{Rng, RngCore};

/// 某层的神经元个数
#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    /// 神经网络由多层组成
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);
        // let mut built_layers = Vec::new();
        // for i in 0..layers.len() - 1 {
        //     let input_size = layers[i].neurons;
        //     // 当前层的输出个数是下一层的输入个数
        //     let output_size = layers[i + 1].neurons;
        //     built_layers.push(Layer::random(input_size, output_size));
        // }

        let layers = layers
            .windows(2)
            .map(|ls| Layer::random(rng, ls[0].neurons, ls[1].neurons))
            .collect();
        Self { layers }
    }
    ///
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // 每一层进行传播计算
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}
#[derive(Debug)]
struct Layer {
    /// 每一层由多个神经元组成
    neurons: Vec<Neuron>,
}

impl Layer {
    fn random(rng: &mut dyn RngCore, input_size: usize, output_size: usize) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::random(rng, input_size))
            .collect();
        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // 该层的每个神经元的输入参数相同
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

/// 每个神经元由 bias 和 权重 组成
#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..=1.0)).collect();
        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());
        let mut output = inputs
            .into_iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();
        output += self.bias;
        // Relu
        output.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    #[test]
    fn random() {
        // 每次都使用相同的种子，所以 rng 会产生相同的随机数
        let mut rng = ChaCha8Rng::from_seed(Default::default());
        let neuron = Neuron::random(&mut rng, 4);
        assert_relative_eq!(neuron.bias, -0.6255188);
        assert_relative_eq!(
            neuron.weights.as_ref(),
            [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
        )
    }

    #[test]
    fn propagate() {
        let neuron = Neuron {
            bias: 0.5,
            weights: vec![-0.3, 0.8],
        };

        // Ensures `.max()` (our ReLU) works:
        assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

        assert_relative_eq!(
            neuron.propagate(&[0.5, 1.0]),
            (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
        );
    }
}
