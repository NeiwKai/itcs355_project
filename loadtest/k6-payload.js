// Usage:
//   k6 run -e TARGET=https://itcs355-project.onrender.com/predict -e PAD_BYTES=0      loadtest/k6-payload.js
//   k6 run -e TARGET=https://itcs355-project.onrender.com/predict -e PAD_BYTES=1000   loadtest/k6-payload.js
//   k6 run -e TARGET=https://itcs355-project.onrender.com/predict -e PAD_BYTES=10000  loadtest/k6-payload.js
//   k6 run -e TARGET=https://itcs355-project.onrender.com/predict -e PAD_BYTES=100000 loadtest/k6-payload.js
//
// PAD_BYTES adds inert bytes to test transport/serialization overhead.
// NOTE: If your FastAPI Pydantic schema uses `extra='forbid'`, adding unknown
// fields like `_pad` will trigger 422 validation errors. Ensure your service
// either allows extra fields or handles padding appropriately.
// LATENCY TARGET — committed 2026-10-09, BEFORE any measurement was taken:

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('predict_latency_ms');
const failures = new Rate('predict_failures');
const padBytes = Number(__ENV.PAD_BYTES || 0);

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '30s',
  thresholds: {
    'predict_latency_ms': ['p(95)<300'],
  },
};

function buildPayload(pad) {
  const body = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.70,
    'citric acid': 0.00,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4,
  };

  if (pad > 0) {
    body['_pad'] = 'x'.repeat(pad);
  }

  return JSON.stringify(body);
}

const payload = buildPayload(padBytes);

const headers = { 'Content-Type': 'application/json' };
const token = __ENV.TOKEN || __ENV.API_SECRET_KEY;
if (token) {
  headers['Authorization'] = `Bearer ${token}`;
}

const targetUrl = __ENV.TARGET || 'https://itcs355-project.onrender.com/predict';

export default function () {
  const res = http.post(targetUrl, payload, { headers });
  latency.add(res.timings.duration);
  failures.add(res.status !== 200);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'quality_score present': (r) => r.status === 200 && r.json('quality_score') !== undefined,
  });
}
