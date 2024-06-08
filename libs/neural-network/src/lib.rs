#[derive(Debug)]
pub struct Network {
    /// 神经网络由多层组成
    layers: Vec<Layer>,
}
impl Network {
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

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
