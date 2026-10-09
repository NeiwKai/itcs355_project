//   k6 run -e TARGET=https://itcs355-project.onrender.com/predict -e VUS=10 loadtest/k6.js
//
// Run this at THREE concurrency levels (1, 10, 50) and record p50, p95, p99,
// throughput, and error rate for each.
//
// LATENCY TARGET — committed 2026-10-09, BEFORE any measurement was taken:
//   p95 < 300ms, error rate < 1%, measured at each of VUS=1,10,50.
//   Rationale: lightweight sklearn-class model, simple JSON in/out, no batch
//   or GPU work on the hot path — 200ms leaves generous headroom for network
//   + serialization overhead on a small cloud instance.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const latency = new Trend('predict_latency_ms');
const failures = new Rate('predict_failures');

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '60s',
  thresholds: {
    'predict_latency_ms': ['p(95)<300'],
    'predict_failures': ['rate<0.01'],
  },
};

const payload = JSON.stringify({
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
});

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
    'version reported': (r) => r.headers['X-Model-Version'] !== undefined || r.headers['x-model-version'] !== undefined,
  });
}
