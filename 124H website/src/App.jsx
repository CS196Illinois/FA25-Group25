// src/App.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import * as tf from "@tensorflow/tfjs";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = "https://api.exchangerate.host"; // exchangerate.host timeseries endpoint used below.  [oai_citation:1‡ExchangeRate Host](https://exchangerate.host/documentation?utm_source=chatgpt.com)

function formatDate(d) {
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

async function fetchRates(base = "USD", symbol = "EUR", startDate, endDate) {
  // 使用 timeseries endpoint 获取区间历史数据
  // Example: GET /timeseries?start_date=2022-01-01&end_date=2022-12-31&base=USD&symbols=EUR
  const url = `${API_BASE}/timeseries`;
  const resp = await axios.get(url, {
    params: {
      start_date: startDate,
      end_date: endDate,
      base,
      symbols: symbol,
      // places: 6, // 可选
    },
  });
  if (!resp.data || !resp.data.rates) throw new Error("API response error");
  // 返回按日期升序的数组 [{date, rate}, ...]
  const rates = Object.entries(resp.data.rates)
    .map(([date, obj]) => ({ date, rate: obj[symbol] }))
    .sort((a, b) => (a.date > b.date ? 1 : -1));
  return rates;
}

function createSequences(dataArr, windowSize = 20) {
  // dataArr: array of numbers (rates)
  // 输出 {xs: Tensor2d, ys: Tensor2d} 适合 LSTM (samples, timesteps, features)
  const xs = [];
  const ys = [];
  for (let i = 0; i + windowSize < dataArr.length; i++) {
    const seq = dataArr.slice(i, i + windowSize);
    const target = dataArr[i + windowSize];
    xs.push(seq);
    ys.push([target]);
  }
  return { xs: tf.tensor2d(xs), ys: tf.tensor2d(ys) };
}

function normalizeSeries(arr) {
  // 简单的 MinMax 归一化到 [0,1]
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const norm = arr.map((v) => (v - min) / (max - min));
  return { norm, min, max };
}
function denormalize(value, min, max) {
  return value * (max - min) + min;
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]); // [{date, rate}]
  const [predictions, setPredictions] = useState([]); // predicted values (dates + rate)
  const [status, setStatus] = useState("");
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    // 启动时自动加载最近 1 年数据并训练（示例）
    runDemo();
    // eslint-disable-next-line
  }, []);

  async function runDemo() {
    setLoading(true);
    setStatus("Fetching data...");
    try {
      const end = new Date();
      const start = new Date();
      start.setFullYear(end.getFullYear() - 1); // 1 year history
      const startDate = formatDate(start);
      const endDate = formatDate(end);
      const rates = await fetchRates("USD", "EUR", startDate, endDate);
      setHistory(rates);
      setStatus("Preprocessing...");
      // extract numeric
      const arr = rates.map((r) => r.rate);
      // normalize
      const { norm, min, max } = normalizeSeries(arr);

      // create sequences
      const WINDOW = 30; // 可调
      const step = 1;
      const xs = [];
      const ys = [];
      for (let i = 0; i + WINDOW < norm.length; i += step) {
        xs.push(norm.slice(i, i + WINDOW));
        ys.push([norm[i + WINDOW]]);
      }
      // Convert to tensors shaped [samples, timesteps, features=1]
      const xsTensor = tf.tensor(xs).expandDims(2); // [N, WINDOW, 1]
      const ysTensor = tf.tensor(ys); // [N,1]

      setStatus("Building model...");
      // Build model
      const model = tf.sequential();
      model.add(
        tf.layers.lstm({
          units: 64,
          inputShape: [WINDOW, 1],
          returnSequences: false,
        })
      );
      model.add(tf.layers.dense({ units: 32, activation: "relu" }));
      model.add(tf.layers.dense({ units: 1, activation: "linear" }));
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "meanSquaredError",
        metrics: ["mse"],
      });

      setStatus("Training (in-browser) - this may take a bit...");
      // Train
      const historyTf = await model.fit(xsTensor, ysTensor, {
        epochs: 30,
        batchSize: 32,
        validationSplit: 0.1,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setStatus(`Epoch ${epoch + 1} - loss: ${logs.loss.toFixed(6)} val_loss: ${logs.val_loss?.toFixed(6)}`);
            await tf.nextFrame();
          },
        },
      });

      setModelInfo({
        summary: model.toString(),
        trainHistory: historyTf.history,
      });

      // Predict next 7 days (iterative forecasting)
      setStatus("Forecasting next 7 days...");
      let recent = norm.slice(-WINDOW); // last window
      const preds = [];
      for (let i = 0; i < 7; i++) {
        const input = tf.tensor(recent).reshape([1, WINDOW, 1]);
        const predTensor = model.predict(input);
        const predVal = (await predTensor.data())[0];
        predTensor.dispose();
        // push predicted normalized, then append to recent
        preds.push(predVal);
        recent = recent.slice(1);
        recent.push(predVal);
      }
      // denormalize preds
      const denormPreds = preds.map((p) => denormalize(p, min, max));
      // build date labels
      const lastDate = new Date(rates[rates.length - 1].date);
      const predPoints = denormPreds.map((val, idx) => {
        const d = new Date(lastDate);
        d.setDate(d.getDate() + idx + 1);
        return { date: formatDate(d), rate: val };
      });
      setPredictions(predPoints);

      // cleanup
      xsTensor.dispose();
      ysTensor.dispose();
      setStatus("Done.");
    } catch (e) {
      console.error(e);
      setStatus("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  // prepare chart data
  const labels = history.map((h) => h.date).concat(predictions.map((p) => p.date));
  const historyRates = history.map((h) => h.rate);
  const predRates = new Array(history.length - 1).fill(null).concat(predictions.map((p) => p.rate));

  const data = {
    labels,
    datasets: [
      {
        label: "历史汇率 (USD→EUR)",
        data: historyRates.concat(new Array(predictions.length).fill(null)),
        tension: 0.2,
      },
      {
        label: "模型预测",
        data: new Array(historyRates.length).fill(null).concat(predictions.map((p) => p.rate)),
        tension: 0.2,
      },
    ],
  };

  return (
    <div style={{ padding: 20, maxWidth: 1100, margin: "0 auto" }}>
      <h2>React + TensorFlow.js — 货币汇率预测示例（USD → EUR）</h2>
      <p>数据源：exchangerate.host 的 timeseries endpoint（免费、含历史数据）。 [oai_citation:2‡ExchangeRate Host](https://exchangerate.host/documentation?utm_source=chatgpt.com)</p>
      <div style={{ marginBottom: 12 }}>
        <button onClick={runDemo} disabled={loading}>
          {loading ? "运行中..." : "重新拉取并训练（最近1年）"}
        </button>
        <span style={{ marginLeft: 12 }}>{status}</span>
      </div>

      <div style={{ height: 420 }}>
        <Line data={data} />
      </div>

      <div style={{ marginTop: 18 }}>
        <h3>最近数据（最后 5）</h3>
        <ul>
          {history.slice(-5).map((h) => (
            <li key={h.date}>
              {h.date}: {h.rate.toFixed(6)}
            </li>
          ))}
        </ul>

        <h3>模型预测（下一周）</h3>
        <ul>
          {predictions.map((p) => (
            <li key={p.date}>
              {p.date}: {p.rate.toFixed(6)}
            </li>
          ))}
        </ul>
      </div>

      <div style={{ marginTop: 16 }}>
        <h4>说明与建议（短）</h4>
        <ol>
          <li>浏览器端训练方便、无后端依赖，但受限于 CPU/GPU 和内存 — 仅适合原型或小规模训练。</li>
          <li>exchangerate.host 提供免费历史与 timeseries 接口；若需要更高频（分钟级）或更可靠的 SLA，可考虑 Alpha Vantage / Fixer 等（可能需 API key 或付费）。 [oai_citation:3‡ExchangeRate Host](https://exchangerate.host/documentation?utm_source=chatgpt.com)</li>
          <li>模型很简单（示例 LSTM），要用于生产请做更多特征工程（宏观变量、技术指标、波动率、新闻等）和更严格的评估（回测、滑点、分布偏差）。</li>
        </ol>
      </div>

      <pre style={{ whiteSpace: "pre-wrap", marginTop: 12, background: "#f7f7f7", padding: 12 }}>
        {modelInfo ? JSON.stringify(modelInfo.trainHistory, null, 2) : "模型训练历史会在此处显示"}
      </pre>
    </div>
  );
}

