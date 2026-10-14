const {spawn} = require('child_process');

const py = spawn('python', ['main.py']);

const testData = {type:'hello',value:'100'};
py.stdin.write(JSON.stringify(testData)+ '\n');

py.stdout.on('data',(data)=>{
    console.log('respond:',data.toString());
});