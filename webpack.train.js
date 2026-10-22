const path = require('path');

module.exports = {
  mode: 'production',
  target: 'node',
  entry: './src/resources/js/physics.js',
  output: {
    filename: 'physics.cjs',
    path: path.resolve(__dirname, 'dist-train'),
    library: { type: 'commonjs2' },
  },
};
