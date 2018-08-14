import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'
import labels from './labels.json'

const ASSETS_URL = `${window.location.origin}/assets`
const MODEL_URL = `${ASSETS_URL}/mobilenet-v2/tensorflowjs_model.pb`
const WEIGHTS_URL = `${ASSETS_URL}/mobilenet-v2/weights_manifest.json`
const IMAGE_SIZE = 224 // Model input size

const loadModel = async () => {
  const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL)
  const input = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])
   // Warm up GPU
  // model.predict({ input }) // MobileNet V1
  model.predict({ Placeholder: input }) // MobileNet V2
  return model
}

const predict = async (img, model) => {
  const t0 = performance.now()
  const image = tf.fromPixels(img).toFloat()
  const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE])
  const offset = tf.scalar(255 / 2)
  const normalized = resized.sub(offset).div(offset)
  const input = normalized.expandDims(0)
  // const output = await tf.tidy(() => model.predict({ input })).data() // MobileNet V1
  const output = await tf.tidy(() => model.predict({ Placeholder: input })).data() // MobileNet V2
  const predictions = labels
    .map((label, index) => ({ label, accuracy: output[index] }))
    .sort((a, b) => b.accuracy - a.accuracy)
  const time = `${(performance.now() - t0).toFixed(1)} ms`
  return { predictions, time }
}

const start = async () => {
  const input = document.getElementById('input')
  const output = document.getElementById('output')
  const model = await loadModel()
  const predictions = await predict(input, model)
  output.append(JSON.stringify(predictions, null, 2))
}

start()
