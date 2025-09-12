document.getElementById('FileInput').addEventListener('change', handleFileUpload);
const preds = [];
/* const labelNames = [
	"light", 
	"moderate-vigorous", 
	"sedentary", 
	"sleep"
];
const labelColors = {
  "light":     						"#009E73",  // Green
  "moderate-vigorous":    "#FF0000",  // Red
  "sedentary":   					"#FFB000",  // Yellow
  "sleep": 								"#648FFF"   // Blue  			
}; */
const labelNames = [
	"WALKING",
	"WALKING_UPSTAIRS",
	"WALKING_DOWNSTAIRS",
	"SITTING",
	"STANDING",
	"LAYING"
];
const labelColors = {
  "WALKING":     						"#009E73",  // Green
  "WALKING_UPSTAIRS":  			"#FF0000",  // Red
  "WALKING_DOWNSTAIRS":   	"#FFB000",  // Yellow
  "SITTING": 								"#648FFF",  // Blue
	"STANDING": 							"#FF7F0E",  // Orange
	"LAYING":   							"#800080"   // Purple
};
const n_outputs = labelNames.length;
const expectedHeader = [
  'acc_x', 'acc_y', 'acc_z',
];
const ws = 2.56;
const sample_freq = 50;
const window_size = 128;
const step_size = window_size;
const n_channels = 3;
async function handleFileUpload(event) {
  preds.length = 0; // Clear old predictions
	
  const file = event.target.files[0];
  if (!file) return;

  const text = await file.text();
  const lines = text.trim().split('\n');

  // Validate header
  const header = lines[0].split(',').map(h => h.trim());

  const headerIsValid = expectedHeader.every((val, i) => val === header[i]);
  if (!headerIsValid) {
    alert("CSV header does not match expected format:\n" + expectedHeader.join(', '));
    return;
  }
  
  // Parse rows after header
  const rows = lines.slice(1).map(row => row.split(',').map(Number));

  if (rows.length < window_size || rows[0].length !== n_channels) {
    alert("CSV must have at least " + window_size + " rows and " + n_channels + " columns.");
    return;
  }
		
  const segments = doSegmentation(rows, windowSize = window_size, stepSize = window_size);
  //for (let i = 0; i < segments.length; i++) {
  //    const segment = segments[i];
	//  const pred = doClassify(segment.flat()); // Classify via WASM
	//  preds.push({ time: i * step_size, label: pred });
  //}
	async function doClassifyAsync(segments) {
		const BATCH_SIZE = Math.floor(60/ws);
		// Process in 1min batches
		for (let i = 0; i < segments.length; i += BATCH_SIZE) {
			const batchEnd = Math.min(i + BATCH_SIZE, segments.length);		
			for (let j = i; j < batchEnd; j++) {
				const segment = segments[j];
				const pred = doClassify(segment.flat()); // Classify via WASM
				preds.push({ time: j * step_size, label: pred });
			}		
			// Update progress and yield control to browser
			//console.log(`${batchEnd} / ${segments.length}`);
			const percent = Math.round((batchEnd / segments.length) * 100);
			progressBar.value = percent;
			progressText.textContent = `${batchEnd} / ${segments.length} (${percent}%)`;
			await new Promise(resolve => setTimeout(resolve, 0));
		}
	}
  await doClassifyAsync(segments);
  drawSignal(rows);

}

// Function adapted from:
// Copyright (c) 2025 Kristof Van Laerhoven
// Licensed under the MIT License.
// Source: https://github.com/kristofvl/wesadviz/
function drawSignal(rows) {
  // Convert data to uPlot format for plotting
  const timeArray = rows.map((_, i) => i);
  const acc_x = rows.map(row => row[0]);
  const acc_y = rows.map(row => row[1]);
  const acc_z = rows.map(row => row[2]);
  const accArray = [acc_x, acc_y, acc_z];

  // Draw hook
  const drawHk = [
    (u) => {
      for (let i = 0; i < preds.length; i++) {
        const pred = preds[i];
        const startTime = pred.time;
        const endTime = (i + 1 < preds.length) ? preds[i + 1].time : pred.time + step_size;
        const fill = labelColors[labelNames[pred.label]] || "gray"; 
        const startPos = u.valToPos(startTime, "x", true);
        const endPos = u.valToPos(endTime, "x", true);
        const width = endPos - startPos;
        u.ctx.fillStyle = fill;
        u.ctx.globalAlpha = 0.25;
        u.ctx.fillRect(startPos, u.bbox.top, width, u.bbox.height);
      }
      u.ctx.globalAlpha = 1.0;
      u.ctx.textAlign = "left";
      u.ctx.fillStyle = "black";
      u.ctx.fillText("accel", 2, 15);
      u.ctx.fillText("acc_x", 2, u.valToPos(acc_x[0], "y", true));
      u.ctx.fillText("acc_y", 2, u.valToPos(acc_y[0], "y", true)); 
      u.ctx.fillText("acc_z", 2, u.valToPos(acc_z[0], "y", true));
    }
  ];
  // wheel scroll zoom
  const wheelZoomHk = [
    (u) => {
      let rect = u.over.getBoundingClientRect();
      u.over.addEventListener(
        "wheel",
        (e) => {
          let oxRange = u.scales.x.max - u.scales.x.min;
          let nxRange = e.deltaY < 0 ? oxRange * 0.95 : oxRange / 0.95;
          let nxMin =
            u.posToVal(u.cursor.left, "x") -
            (u.cursor.left / rect.width) * nxRange;
          if (nxMin < 0) nxMin = 0;
          let nxMax = nxMin + nxRange;
          if (nxMax > u.timeArray[timeArray.length - 1])
            nxMax = u.timeArray[timeArray.length - 1];
          u.setScale("x", { min: nxMin, max: nxMax });
        },
        { passive: true }
      );
    }
  ];
	const drawClearHk = [
		(u) => {
		},
	];
  const opts = {
    width: 1280,
    height: 280,
    cursor: {
      y: false,
    },
    axes: [
      { label: "Time" },
      { label: "Signal", scale: "y", side: 1, grid: { show: true } }
    ],
    scales: {
      x: { time: false },
      y: { auto: true }
    },
    series: [
      { label: "time" },
      { 
        label: "total_acc_x", 
        stroke: "red",
        width: 1
      },
      { 
        label: "total_acc_y", 
        stroke: "yellow", 
        width: 1
      },
      { 
        label: "total_acc_z", 
        stroke: "blue",
        width: 1
      }
    ],
    hooks: {
      draw: [...drawHk],
      drawClear: drawClearHk,
      ready: wheelZoomHk
    },
    legend: { show: false }
  };

  const div = document.querySelector("#draw");
  while (div.firstChild) div.removeChild(div.firstChild);
  const plot = new uPlot(opts, [timeArray, ...[acc_x, acc_y, acc_z]], div);
  
  // Simple manual legend
  const legendDiv = document.createElement("div");
  legendDiv.id = "legend";
  legendDiv.style.marginTop = "10px";
  legendDiv.innerHTML = labelNames.map(label => `
    <span style="margin-right: 16px;">
      <span style="
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: ${labelColors[label]};
        opacity: 0.25;
        border: 1px solid #000;
        margin-right: 4px;
      "></span>${label}
    </span>
  `).join("");
  // Append new legend
  document.querySelector("#draw").appendChild(legendDiv);
}

function drawHist(classify) {
  const t = 5.0;
  const maxC = classify.reduce((a, b) => a > b ? a : b);
  const plot = Plot.plot({
    marginTop: 50,
    width: 280,
    height: 200,
    axis: null,
    x: { axis: "bottom" },
    y: { domain: [0, 1] },
    marks: [
      Plot.text(['Classification'], { frameAnchor: "Top", dy: -25 }),
      Plot.barY(classify.map((v) => Math.exp(v / t) / Math.exp(maxC / t)), { fill: 'SteelBlue' })
    ],
  });

  const div = document.querySelector("#hist");
  while (div.firstChild) {
    div.removeChild(div.lastChild);
  }
  div.append(plot);
}

let conv1_w_ptr = -1, conv1_b_ptr = -1;
let conv2_w_ptr = -1, conv2_b_ptr = -1;
let conv3_w_ptr = -1, conv3_b_ptr = -1;
let fc1_w_ptr = -1, fc1_b_ptr = -1;
let fc2_w_ptr = -1, fc2_b_ptr = -1;
function doLoadData() {
  (async () => {
    try {

      // Load conv1 weights
      let response = await fetch("./conv1_weight.data");
      if (!response.ok) throw new Error(`Problem downloading conv1_weight (${response.status})`);
      let bytes = await response.arrayBuffer();
      let floats = new Float32Array(bytes);
      conv1_w_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv1_w_ptr / 4);
			//console.log("len:", floats.length);
			//console.log("first 10:", floats.slice(0,10)); check against test.py

      // Load conv1 bias
      response = await fetch("./conv1_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv1_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      conv1_b_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv1_b_ptr / 4);

      // Load conv2 weights
      response = await fetch("./conv2_weight.data");
      if (!response.ok) throw new Error(`Problem downloading conv2_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      conv2_w_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv2_w_ptr / 4);

      // Load conv2 bias
      response = await fetch("./conv2_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv2_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      conv2_b_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv2_b_ptr / 4);
			
      // Load conv3 weights
      response = await fetch("./conv3_weight.data");
      if (!response.ok) throw new Error(`Problem downloading conv3_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      conv3_w_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv3_w_ptr / 4);

      // Load conv3 bias
      response = await fetch("./conv3_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv3_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      conv3_b_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, conv3_b_ptr / 4);

      // Load fc1 weights
      response = await fetch("./fc1_weight.data");
      if (!response.ok) throw new Error(`Problem downloading fc1_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      fc1_w_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, fc1_w_ptr / 4);

      // Load fc1 bias
      response = await fetch("./fc1_bias.data");
      if (!response.ok) throw new Error(`Problem downloading fc1_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      fc1_b_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, fc1_b_ptr / 4);

      // Load fc2 weights
      response = await fetch("./fc2_weight.data");
      if (!response.ok) throw new Error(`Problem downloading fc2_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      fc2_w_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, fc2_w_ptr / 4);

      // Load fc2 bias
      response = await fetch("./fc2_bias.data");
      if (!response.ok) throw new Error(`Problem downloading fc2_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      floats = new Float32Array(bytes);
      fc2_b_ptr = Module._malloc(floats.byteLength);
      Module.HEAPF32.set(floats, fc2_b_ptr / 4);

      // Ready to start

    } catch (err) {
      console.error(err);
    }
  })();
}

function doSegmentation(data, windowSize = 128, stepSize = 128) {
  const segments = [];
  for (let start = 0; start + windowSize <= data.length; start += stepSize) {
    const segment = data.slice(start, start + windowSize);
    segments.push(segment);
  }
  return segments;
}

function doClassify(sequence) {
  if (
    conv1_w_ptr < 0 || conv1_b_ptr < 0 ||
    conv2_w_ptr < 0 || conv2_b_ptr < 0 ||
		conv3_w_ptr < 0 || conv3_b_ptr < 0 ||
    fc1_w_ptr < 0 || fc1_b_ptr < 0 ||
		fc2_w_ptr < 0 || fc2_b_ptr < 0 
  ) {
    throw new Error("Can't classify input. Weights not downloaded yet.");
  }

  // Copy data to contiguous Wasm memory
  const sequence_ptr = Module._malloc(window_size * n_channels * 4);
  Module.HEAPF32.set(sequence, sequence_ptr / 4); //Module.HEAPF32 and 64 are 1D flat typed arrays

  // Call Wasm classifier and store results
  const out_ptr = Module._malloc(n_outputs*4);
  Module._classifier(conv1_w_ptr, conv1_b_ptr, conv2_w_ptr, conv2_b_ptr, conv3_w_ptr, conv3_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr, sequence_ptr, out_ptr);

  // Draw results as a histogram
  const out_bytes = Module.HEAPF32.subarray(out_ptr / 4, (out_ptr / 4) + n_outputs);
  //console.log("Output values:", out_bytes);
  //drawHist(out_bytes);
  
  var output = softmax(out_bytes, t=n_outputs);
  output = output.indexOf(Math.max(...output));

  // Wasm memory cleanup
  Module._free(sequence_ptr);
  Module._free(out_ptr);
  
  return output;
}
function softmax(arr, t = 2.0) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x / t - max / t));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map(e => e / sum);
}
