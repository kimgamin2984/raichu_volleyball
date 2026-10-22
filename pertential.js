const { spawn } = require('child_process');

// 1. 파이썬 뇌 켜기
const py = spawn('python', ['main.py']);
let buffer = '';

// ==========================================
// 2. 파이썬 전송 헬퍼 함수
// ==========================================
function sendAct(state) {
    py.stdin.write(JSON.stringify({ type: 'act', state: state }) + '\n');
}

function sendObserve(state, action, reward, nextState, done) {
    const payload = { type: 'observe', state, action, reward, next_state: nextState, done };
    py.stdin.write(JSON.stringify(payload) + '\n');
}

// ==========================================
// 3. 환경(Environment) 설정 및 상태 추출
// ==========================================
const HALF_WIDTH = 216;
const HALF_HEIGHT = 152;

const normX = (x) => (x - HALF_WIDTH) / HALF_WIDTH;
const normY = (y) => (y - HALF_HEIGHT) / HALF_HEIGHT;

function extractState() {
    // ★ 실제 physics.js의 객체명에 맞게 매핑하세요
    return [
        normX(player1.x), normY(player1.y),
        normX(player2.x), normY(player2.y),
        normX(ball.x), normY(ball.y),
        ball.vx || 0, ball.vy || 0, // 공의 속도 (없으면 임시 0)
        0, 0 // 총 10개 구색 맞추기용
    ];
}

// ★ 파이썬 행동(0~31)을 키보드 입력으로 바꾸는 함수
function applyAction(action) {
    // 예: action 값에 따라 비트 연산으로 키 상태(isLeft, isRight 등) 변경
    // (서포트 개발자가 이 부분을 실제 게임 입력 로직으로 채워야 합니다)
}

// ★ 1프레임 지난 후 보상(+100, -100)과 세트 종료 여부 판정
function checkRewardAndDone() {
    let reward = 0;
    let done = false;
    
    // 예시: 공이 상대방 땅(우측)에 닿아서 우리 득점
    // if (공 바닥 충돌 && 우측 진영) { reward = 100; done = true; }
    // 예시: 공이 우리 땅(좌측)에 닿아서 실점
    // if (공 바닥 충돌 && 좌측 진영) { reward = -100; done = true; }

    return { reward, done };
}

// ==========================================
// 4. 게임 루프 해킹 (자동 루프 무력화 및 수동화)
// ==========================================
let originalUpdate = null;

if (typeof updateGame !== 'undefined') {
    originalUpdate = updateGame;
    // 게임 엔진이 자동으로 부르는 updateGame을 빈 함수로 덮어써서 기절시킴
    updateGame = function() { }; 
}

// 핑퐁 추적용 메모리
let lastState = null;
let lastAction = null;

// ==========================================
// 5. 무한 핑퐁 수신부 (핵심 학습 루프)
// ==========================================
py.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop(); // 안 끝난 조각은 킵

    for (const line of lines) {
        if (!line.trim()) continue;

        try {
            const res = JSON.parse(line);

            // [A] 파이썬이 "이 행동 해!" 라고 줬을 때
            if (res.action !== undefined) {
                lastAction = res.action;

                // 1. 게임에 키보드 입력 적용
                applyAction(lastAction);

                // 2. 수동으로 딱 1프레임만 진행! (핵심)
                if (originalUpdate) originalUpdate();

                // 3. 1프레임 진행 후의 결과 확인
                const nextState = extractState();
                const { reward, done } = checkRewardAndDone();

                // 4. 파이썬에게 결과 보고 (여기서 파이썬 뇌가 학습함)
                sendObserve(lastState, lastAction, reward, nextState, done);
            } 
            
            // [B] 파이썬이 "학습/저장 완료했으니 다음 상태 내놔" 라고 줬을 때
            else if (res.status) {
                // console.log(`[학습 중] 상태: ${res.status} | Loss: ${res.loss}`);
                
                // 1. 만약 라운드가 끝났다면(done), 환경 초기화(공 리셋 등) 로직을 여기에 넣습니다.
                // if (방금 보낸 데이터가 done == true 였다면) { env.reset(); }

                // 2. 최신 상태 뽑아서 파이썬에게 다음 행동 질문
                lastState = extractState();
                sendAct(lastState);
            }

        } catch (err) {
            console.error('[JSON 에러]', err.message, line);
        }
    }
});

py.stderr.on('data', (data) => console.error('[Python Error]:', data.toString()));

// ==========================================
// 6. 엔진 시동 (최초 1회 발사)
// ==========================================
setTimeout(() => {
    console.log('[System] AI 피카츄 배구 턴제 학습을 시작합니다!');
    lastState = extractState();
    sendAct(lastState);
}, 2000); // 게임 화면 렌더링 2초 기다렸다가 핑퐁 시작