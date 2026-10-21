const HALF_WIDTH = 216;  // 가로 432 / 2
const HALF_HEIGHT = 152; // 세로 304 / 2

// 1. 좌표 정규화 함수 (-1.0 ~ 1.0)
const normX = (x) => (x - HALF_WIDTH) / HALF_WIDTH;
const normY = (y) => (y - HALF_HEIGHT) / HALF_HEIGHT;

// 2. State 추출 함수 (원하시는 숫자 10개 배열로 구성)
function extractState() {
  // ※ physics.js 내 실제 객체명(player1, player2, ball 등)에 맞춰 적용
  return [
    normX(player1.x), normY(player1.y), // P1 좌표
    normX(player2.x), normY(player2.y), // P2 좌표
    normX(ball.x), normY(ball.y),       // 공 좌표
    // 공의 속도(vx, vy)나 플레이어 상태 등 나머지 4개를 추가해 10개 완성
  ];
}

// 3. 게임 루프에 래핑(Wrapping) 적용
if (typeof updateGame !== 'undefined') {
  const originalUpdate = updateGame;

  updateGame = function() {
    originalUpdate(); // 원본 1프레임 먼저 실행
    
    const currentState = extractState(); // 환경 State 추출
    
    // TODO: currentState를 파이썬 AI로 전송
  };
}