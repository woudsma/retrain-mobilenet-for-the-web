import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'
import labels from './labels.json'

const ASSETS_URL = `${window.location.origin}/assets`
const MODEL_URL = `${ASSETS_URL}/model/tensorflowjs_model.pb`
const WEIGHTS_URL = `${ASSETS_URL}/model/weights_manifest.json`
const IMAGE_SIZE = 128 // Model input size

const loadModel = async () => {
  const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL)
  const input = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])
  model.predict({ input }) // Warm up GPU
  return model
}

const predict = async (img, model) => {
  const t0 = performance.now()
  const image = tf.fromPixels(img).toFloat()
  const resized = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE])
  const offset = tf.scalar(255 / 2)
  const normalized = resized.sub(offset).div(offset)
  const input = normalized.expandDims(0)
  const output = await tf.tidy(() => model.predict({ input })).data()
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
