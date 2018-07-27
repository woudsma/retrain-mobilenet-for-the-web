# Retrained MobileNet V1 TensorFlow.js example app

Repository containing a HTML/JS boilerplate for serving a retrained MobileNet V1 model.  
Also includes a `Dockerfile` and `nginx.default.conf` (with gzip enabled) for easy deploys on e.g. [Dokku](http://dokku.viewdocs.io/dokku/).  

#### Installation
```
git clone https://github.com/woudsma/retrained-mobilenet-v1-tfjs-example
cd retrained-mobilenet-v1-tfjs-example

# Install dependencies
npm install

# Start development server
npm start

# Create production build
npm run build
```

----

# Retrain a MobileNet model for the web
###### *DRAFT*  
Combining [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0) and [TensorFlow.js](https://github.com/tensorflow/tfjs).  
Retrain a MobileNet V1 model on your own dataset using the CPU only.  
I'm using a MacBook Pro without Nvidia GPU.  

[MobileNets](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) can be used for image classification. This guide shows the steps I took to retrain a MobileNet on a custom dataset, and how to convert and use the retrained model in the browser using TensorFlow.js. The total time to set up, retrain the model and use it in the browser can take less than 45 minutes (depending on the size of your dataset).  

Repository containing the example app (HTML/JS and a retrained MobileNet V1 model).  
[https://github.com/woudsma/retrain-mobilenet-v1-for-the-web](https://github.com/woudsma/retrain-mobilenet-v1-for-the-web)

---

## 1. Python setup
Set up a virtual environment in Python. This keeps your system clean and dependencies separated. It's good practice not to `sudo` install packages. You can skip this section if you are already familiar with Python and [virtualenv](https://docs.python-guide.org/dev/virtualenvs/).  

We will use [virtualenv-burrito](https://github.com/brainsik/virtualenv-burrito), which is a script for installing both virtualenv and virtualenv-wrapper.
```sh
cd
curl -sL https://raw.githubusercontent.com/brainsik/virtualenv-burrito/master/virtualenv-burrito.sh | $SHELL
source ~/.venvburrito/startup.sh

# Create project environment
mkvirtualenv myproject
deactivate
```
Activate an environment: `workon foo`  
Deactivate current environment: `deactivate`  
Create an environment: `mkvirtualenv foo`  
Remove an environment: `rmvirtualenv foo`  
List available environments: `lsvirtualenv`  

---

## 2. Retrain a MobileNet model using a custom dataset
**If you get stuck at any point, see the [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0) codelab**  

Active the project's virtualenv, install TensorFlow.js, and git clone the  `tensorflow-for-poets-2` [repository](https://github.com/googlecodelabs/tensorflow-for-poets-2).
```sh
# Activate project environment
# Install TensorFlow.js (includes tensorflow, tensorboard, tensorflowjs_converter)
workon myproject
pip install tensorflowjs

# Clone repo
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
```
*Note: all further commands assume the present working directory is `tensorflow-for-poets-2`, and project virtualenv is activated.*  

Run the `scripts/retrain.py` script with the `-h` (help) flag to see all options.  
```sh
# cd /path/to/tensorflow-for-poets-2
python -m scripts.retrain -h
```

#### Add a dataset  
Create folders in `tf_files` and add your dataset (folders containing images) to the `tf_files/dataset` directory. The classification labels used when running inference will be generated from the **folder names**.
```sh
# Create folders
mkdir -p tf_files/{bottlenecks,dataset,models,training_summaries}

# Add your dataset
cp -R /path/to/my-dataset/* tf_files/dataset

# Or use the 'flowers' dataset
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C .
cp -R flower_photos/* tf_files/dataset
```

#### Start TensorBoard  
Start [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) from a new terminal window and visit `http://localhost:6006/`.  
```sh
# Open new terminal window
cd /path/to/tensorflow-for-poets-2
workon myproject
tensorboard --logdir tf_files/training_summaries
```

#### Retrain a model using a pre-trained [MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) model  
To retrain a MobileNet model, choose an architecture from [this page](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html), and run the `scripts/retrain.py` script with your dataset. This guide only covers MobileNet V1. We found that `mobilenet_0.50_224` provides both decent accuracy and acceptable filesize (the model takes ~2.3MB after gzip compression). Smaller models such as `mobilenet_0.25_128` provide lower accuracy but require less bandwidth, and vice versa.  

This will take a few minutes (using the flowers dataset), or longer depending on the size of your dataset.
```sh
# Set environment variables
IMAGE_SIZE=128
ARCHITECTURE=mobilenet_0.25_$IMAGE_SIZE

# Start training
python -m scripts.retrain \
  --image_dir=tf_files/dataset \
  --model_dir=tf_files/models \
  --architecture=$ARCHITECTURE \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --bottleneck_dir=tf_files/bottlenecks \
  --summaries_dir=tf_files/training_summaries/$ARCHITECTURE \
  --how_many_training_steps=400 \
  --learning_rate=0.001
```
*Note: keep an eye on TensorBoard ([http://localhost:6006](http://localhost:6006)) during training.*  

For more information and how to adjust hyperparameters, check out the full [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0) codelab.  

#### Test the model by classifying an image  
Classify an image using the `scripts/label_image.py` script.  
(e.g. `tf_files/dataset/daisy/21652746_cc379e0eea_m.jpg` if you've retrained the model on the flowers dataset).
```sh
python -m scripts.label_image \
  --graph=tf_files/retrained_graph.pb \
  --input_width=$IMAGE_SIZE \
  --input_height=$IMAGE_SIZE \
  --image=tf_files/dataset/daisy/21652746_cc379e0eea_m.jpg  

# Top result should be 'daisy'
```

---

## 3. Optimize for the web

#### Quantize graph
Quantize the retrained graph using the `scripts/quantize_graph.py` script. Although you could use and serve the retrained graph, serving the quantized graph saves bandwidth when using gzip compression.
```sh
python -m scripts.quantize_graph \
  --input=tf_files/retrained_graph.pb \
  --output=tf_files/quantized_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```
*Optional*: compare size of gzipped graphs.
```sh
gzip -k tf_files/retrained_graph.pb tf_files/quantized_graph.pb
du -h tf_files/*.gz

# Clean up
rm tf_files/*.gz
```

#### Convert to TensorFlow.js model
Convert the quantized retrained graph to a TensorFlow.js compatible model using [tfjs-converter](https://github.com/tensorflow/tfjs-converter), and save in a new `tf_files/web` folder.
```sh
tensorflowjs_converter \
  --input_format=tf_frozen_model \
  --output_node_names=final_result \
  tf_files/quantized_graph.pb \
  tf_files/web
```

#### Add labels
Create a JSON file from `retrained_labels.txt` using [jq](https://stedolan.github.io/jq/), this way we can easily import the dataset labels in JavaScript.
```sh
# Install 'jq' once with homebrew (https://brew.sh/)
brew install jq

# Create JSON file from newline-delimited text file
cat tf_files/retrained_labels.txt | jq -Rsc '. / "\n" - [""]' > tf_files/web/labels.json
cat tf_files/web/labels.json
```
*Returns:* `["daisy","dandelion","roses","sunflowers","tulips"]`  

Folder structure after running `tensorflowjs_converter` and converting the dataset labels to `labels.json`.
```
/path/to/tensorflow-for-poets-2/tf_files
├── retrained_graph.pb
├── retrained_labels.txt
├── quantized_graph.pb
├── web
│   ├── group1-shard1of1
│   ├── tensorflowjs_model.pb
│   ├── labels.json
│   └── weights_manifest.json
├── bottlenecks
│   └── ...
├── dataset
│   └── ...
├── models
│   └── ...
└── training_summaries
    └── ...
```
*Optional*: check gzipped TensorFlow.js model size.
```sh
tar -czf tf_files/web.tar.gz tf_files/web
du -h tf_files/web.tar.gz

# Clean up
rm tf_files/web.tar.gz
```

---

## 4. Classifying images in the browser
Create an app to run predictions in the browser using the retrained model converted by `tensorflowjs_converter`.  

With a few lines of code, we can classify an image using the retrained model. In this example, we use an `<img>` element as input to get a prediction. The model should also be able to accept `<video>` and `<canvas>` elements as input.  

#### Prepare app folder structure and install dependencies
###### Or clone the example [repository](https://github.com/woudsma/retrain-mobilenet-v1-for-the-web)
```sh
# Create app folder structure
mkdir -p myproject-frontend/{public/assets/{model,images},src}
cd myproject-frontend

# Copy web model files to assets folder
# Move the labels JSON file into the src folder
cp /path/to/tensorflow-for-poets-2/tf_files/web/* public/assets/model
mv public/assets/model/labels.json src/labels.json

# Create HTML/JS files
touch public/index.html src/index.js

# Install dependencies
npm init -y
npm install react-scripts @tensorflow/tfjs @tensorflow/tfjs-core @tensorflow/tfjs-converter

# Add a few test images to public/assets/images manually
```
*I'm using the [react-scripts](https://www.npmjs.com/package/react-scripts) package as development server and build tool (used by [create-react-app](https://github.com/facebook/create-react-app)). This saves some time writing webpack configs, etc.*  

App folder structure after setting up.
```
/path/to/myproject-frontend
├── node_modules
│   └── ...
├── package-lock.json
├── package.json
├── public
│   ├── assets
│   │   ├── images
│   │   │   └── some-flower.jpg
│   │   └── model
│   │       ├── group1-shard1of1
│   │       ├── tensorflowjs_model.pb
│   │       └── weights_manifest.json
│   └── index.html
└── src
    ├── index.js
    └── labels.json
```

#### Add HTML / JS
Edit *public/index.html* to:
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Image classifier</title>
  </head>
  <body>
    <img id="input" src="assets/images/some-flower.jpg" />
    <pre id="output"></pre>
  </body>
</html>
```
Edit *src/index.js* to:
```js
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
```
Add to *package.json* scripts:
```js
"scripts": {
  "start": "react-scripts start",
  "build": "react-scripts build"
}
```
#### Run the app
Start the development server, run `npm start`.  
(or `npx react-scripts start`)  
Opens a browser window at [http://localhost:3000](http://localhost:3000)  
Watches project files and auto-reloads browser on change.  

Create production build, run `npm run build`.  
(or `npx react-scripts build`)  
Outputs `build` folder with static assets.  

---

#### Result
Total size: 2MB (using gzip compression).  

![Result](https://i.imgur.com/EXmdJ0V.jpg "Result")  

---

## gzip
It's possible to save bandwidth by serving static assets (including the model) by using [gzip](https://en.wikipedia.org/wiki/Gzip) compression. This can be done manually or by enabling gzip in your server config.  

For example, add to */etc/nginx/conf.d/default.conf*
```nginx
server {
  ...

  gzip on;
  gzip_vary on;
  gzip_static on;
  gzip_types text/plain application/javascript application/octet-stream;
  gzip_min_length 256;
}
```
If you are using CloudFlare CDN, make sure to disable Brotli compression (for some reason it does not serve `application/octet-stream` files with their original gzip compression).  

---

Please let me know if you notice any mistakes or things to improve. I'm always open to suggestions and feedback!  

All credits go to the creators of [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0) and [TensorFlow.js](https://js.tensorflow.org/). This guide is basically a combination of the original TensorFlow for Poets guide and the TensorFlow.js [documentation](https://js.tensorflow.org/tutorials/). Check out [ml5js](https://ml5js.org/).
