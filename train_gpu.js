'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const localPython = path.join(__dirname, process.platform === 'win32'
  ? '.venv/Scripts/python.exe'
  : '.venv/bin/python');
const python = process.env.PIKA_PYTHON ||
  (fs.existsSync(localPython) ? localPython : (process.platform === 'win32' ? 'py' : 'python3'));
const args = python === 'py'
  ? ['-3', '-u', 'train_gpu.py', ...process.argv.slice(2)]
  : ['-u', 'train_gpu.py', ...process.argv.slice(2)];
const worker = spawn(python, args, { cwd: __dirname, stdio: 'inherit' });

worker.on('error', (error) => {
  process.stderr.write('Could not start Python (' + python + '): ' + error.message + '\n');
  process.exitCode = 1;
});
worker.on('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code === null ? 1 : code;
});
