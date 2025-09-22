document.getElementById('FileInput').addEventListener('change', handleFileUpload);
let plot;
const preds = [];
const labelNames = [
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
};
/* const labelNames = [
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
}; */
const n_outputs = labelNames.length;
const expectedHeader = [
  'acc_x', 'acc_y', 'acc_z',
];
const ws = 2.56;
const sample_freq = 50;
const window_size = 128;
const step_size = window_size;
const n_channels = 3;
const HeapTypes = {
  "int8":    { ArrayType: Int8Array,   Heap: () => Module.HEAP8,    n_bytes: 1 },
  "int16":   { ArrayType: Int16Array,  Heap: () => Module.HEAP16,   n_bytes: 2 },
  "float32": { ArrayType: Float32Array,Heap: () => Module.HEAPF32,  n_bytes: 4 },
  "float64": { ArrayType: Float64Array,Heap: () => Module.HEAPF64,  n_bytes: 8 },
};
// global data type
const DATA_TYPE = "float32";
const T = HeapTypes[DATA_TYPE];
// helper func
function allocAndCopy(bytes) {
  let arr = new T.ArrayType(bytes);
  let ptr = Module._malloc(arr.byteLength);
  T.Heap().set(arr, ptr / T.n_bytes);
  return ptr;
}
async function handleFileUpload(event) {
  preds.length = 0; // Clear old predictions
	
  const file = event.target.files[0];
  if (!file) return;

/*   // loading CSV in
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
  } */
		
	// streaming bin file
	const reader = file.stream().getReader();
	const rowByteSize = n_channels * T.n_bytes;
	let unloaded = new Uint8Array(0);
	let rows = [];
	let i=0;
	const BATCH_SIZE = 128;
	while (true) {
		const { value: chunk, done } = await reader.read();
		if (done) break;
		// browser optimized packet size - force into window of BATCH_SIZE rows
		const packetBuffer = new Uint8Array(unloaded.length + chunk.length);
		packetBuffer.set(unloaded, 0);
		packetBuffer.set(chunk, unloaded.length);
		let loaded = 0;
		let windowBuffer = []; //segment
		while (packetBuffer.length - loaded >= rowByteSize) {
			const rowBytes = packetBuffer.slice(loaded, loaded + rowByteSize);
			const row = Array.from(new T.ArrayType(rowBytes.buffer, rowBytes.byteOffset, n_channels));
			windowBuffer.push(row);
			loaded += rowByteSize;
			if (windowBuffer.length === BATCH_SIZE) {
				rows.push(...windowBuffer);
				i+=1;
				//const pred = doClassify(windowBuffer.flat()); // Classify via WASM
				//preds.push({ time: i * step_size, label: pred });
				windowBuffer = [];
			}
		}
		unloaded = packetBuffer.slice(loaded); //overload
		if (windowBuffer.length > 0) rows.push(...windowBuffer);
	}
	if (rows.length < window_size || rows[0].length !== n_channels) {
		alert("BIN must have at least " + window_size + " rows and " + n_channels + " columns.");
		return;
	}

  const segments = doSegmentation(rows, windowSize = window_size, stepSize = window_size);
  //for (let i = 0; i < segments.length; i++) {
  //    const segment = segments[i];
	//    const pred = doClassify(segment.flat()); // Classify via WASM
	//    preds.push({ time: i * step_size, label: pred });
  //}
	
	const scale = 1.0//32767.0;
	const maxVal = 1.0//5.0;
	rows = rows.map(row => row.map(v => v / (scale/maxVal)));
	drawSignal(rows);
	async function doClassifyAsync(segments) {
		const update_plt_flag = document.getElementById("live-update");
		const BATCH_SIZE = 1; //32; //Math.max(1, Math.floor(segments.length/100));
    for (let i = 0; i < segments.length; i += BATCH_SIZE) {
			// Process in batches
			const batchEnd = Math.min(i+BATCH_SIZE, segments.length);
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
			if (update_plt_flag.checked) {
					plot.redraw();
			}
		}
		if (!update_plt_flag.checked) {
			plot.redraw();
		}
	}
  await doClassifyAsync(segments);
	//drawSignal(rows);
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
          if (nxMax > u.data[0][u.data[0].length - 1])
            nxMax = u.data[0][u.data[0].length - 1];
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
        label: "acc_x", 
        stroke: "red",
        width: 1
      },
      { 
        label: "acc_y", 
        stroke: "yellow", 
        width: 1
      },
      { 
        label: "acc_z", 
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
  plot = new uPlot(opts, [timeArray, ...[acc_x, acc_y, acc_z]], div);
  
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
			conv1_w_ptr = allocAndCopy(bytes);
			//console.log("len:", floats.length);
			//console.log("first 10:", floats.slice(0,10)); check against test.py

      // Load conv1 bias
      response = await fetch("./conv1_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv1_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      conv1_b_ptr = allocAndCopy(bytes);

      // Load conv2 weights
      response = await fetch("./conv2_weight.data");
      if (!response.ok) throw new Error(`Problem downloading conv2_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      conv2_w_ptr = allocAndCopy(bytes);

      // Load conv2 bias
      response = await fetch("./conv2_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv2_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      conv2_b_ptr = allocAndCopy(bytes);
			
      // Load conv3 weights
      response = await fetch("./conv3_weight.data");
      if (!response.ok) throw new Error(`Problem downloading conv3_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      conv3_w_ptr = allocAndCopy(bytes);

      // Load conv3 bias
      response = await fetch("./conv3_bias.data");
      if (!response.ok) throw new Error(`Problem downloading conv3_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      conv3_b_ptr = allocAndCopy(bytes);

      // Load fc1 weights
      response = await fetch("./fc1_weight.data");
      if (!response.ok) throw new Error(`Problem downloading fc1_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      fc1_w_ptr = allocAndCopy(bytes);

      // Load fc1 bias
      response = await fetch("./fc1_bias.data");
      if (!response.ok) throw new Error(`Problem downloading fc1_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      fc1_b_ptr = allocAndCopy(bytes);

      // Load fc2 weights
      response = await fetch("./fc2_weight.data");
      if (!response.ok) throw new Error(`Problem downloading fc2_weight (${response.status})`);
      bytes = await response.arrayBuffer();
      fc2_w_ptr = allocAndCopy(bytes);

      // Load fc2 bias
      response = await fetch("./fc2_bias.data");
      if (!response.ok) throw new Error(`Problem downloading fc2_bias (${response.status})`);
      bytes = await response.arrayBuffer();
      fc2_b_ptr = allocAndCopy(bytes);

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
  const sequence_ptr = Module._malloc(window_size * n_channels * T.n_bytes);
  T.Heap().set(sequence, sequence_ptr / T.n_bytes); //Module.HEAPF32 and 64 are 1D flat typed arrays

  // Call Wasm classifier and store results
  const out_ptr = Module._malloc(n_outputs*T.n_bytes);
  Module._classifier(conv1_w_ptr, conv1_b_ptr, conv2_w_ptr, conv2_b_ptr, conv3_w_ptr, conv3_b_ptr, fc1_w_ptr, fc1_b_ptr, fc2_w_ptr, fc2_b_ptr, sequence_ptr, out_ptr);

  // Draw results as a histogram
  const out_bytes = T.Heap().subarray(out_ptr / T.n_bytes, (out_ptr / T.n_bytes) + n_outputs);
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
