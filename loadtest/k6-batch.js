// Single mode (100 sequential requests per VU loop):
//   k6 run -e BASE=https://itcs355-project.onrender.com -e MODE=single -e VUS=10 loadtest/k6-batch.js
//
// Batch mode (1 request containing 100 rows per VU loop):
//   k6 run -e BASE=https://itcs355-project.onrender.com -e MODE=batch -e VUS=10 loadtest/k6-batch.js
// LATENCY TARGET — committed 2026-10-09, BEFORE any measurement was taken:

import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

const rowLatency = new Trend('per_row_latency_ms');
const batchLatency = new Trend('batch_call_latency_ms');

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '30s',
  thresholds: {
    // 1. Latency target: 95th percentile under 300ms for single requests
    'per_row_latency_ms': ['p(95)<300'],

    // 2. Latency target for 100-row batch requests (generous 1500ms headroom)
    'batch_call_latency_ms': ['p(95)<1500'],
  },
};

// Valid sample matching red wine dataset features
const row = {
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

const rows100 = new Array(100).fill(row);
const mode = __ENV.MODE || 'single';

const headers = { 'Content-Type': 'application/json' };
const token = __ENV.TOKEN || __ENV.API_SECRET_KEY;
if (token) {
  headers['Authorization'] = `Bearer ${token}`;
}

const baseUrl = (__ENV.BASE || 'https://itcs355-project.onrender.com').replace(/\/$/, '');

export default function () {
  if (mode === 'batch') {
    const res = http.post(`${baseUrl}/predict/batch`, JSON.stringify({ rows: rows100 }), { headers });
    batchLatency.add(res.timings.duration);
    check(res, {
      'batch status 200': (r) => r.status === 200,
      'quality_scores array present': (r) => r.status === 200 && Array.isArray(r.json('quality_scores')),
    });
  } else {
    for (let i = 0; i < 100; i++) {
      const res = http.post(`${baseUrl}/predict`, JSON.stringify(row), { headers });
      rowLatency.add(res.timings.duration);
      check(res, { 'single status 200': (r) => r.status === 200 });
    }
  }
}
