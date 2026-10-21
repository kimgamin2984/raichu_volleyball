const {spawn} = require('child_process');

const py = spawn('python', ['main.py']);

const testData = {type:'hello',value:'100'};
py.stdin.write(JSON.stringify(testData)+ '\n');

py.stdout.on('data',(data)=>{
    console.log('respond:',data.toString());
});

function sendAct(state) {
    const data = {type: 'act', state: state};
    py.stdin.write(JSON.stringify(data)+'\n');
}

function sendObserve(state, action, reward, nextState, done) {
    const data = {type: 'observe', state: state, action: action, reward: reward, next_state: nextState, done: done};
    py.stdin.write(JSON.stringify(data)+'\n');
}