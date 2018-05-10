package ab.demo;

import java.awt.Point;

public class InfoClass {
	public static int info_level;
	public static Point info_rp;
	public static double info_ang;
//	public static int info_score;
//	public static String info_state;
}

//column 이름을 가지는 csv를 만들려고 함
//level release_point angle score info_state
//  1     x  y          z     xx     WON
//  2     a  b          c     aa      - 
//  2     i  j          k     ii     LOST 

//우선 문자열로 CSV 만드는게 간단해 보여서 이렇게 했는데, 리스트로 바꿔놓아야 할듯!!
//뭔가 기록이 되긴 하는데, 어디서 기록을 할지를 아직 못찾음...
//그리고 filepath 같은거 좀 깔끔하게 할 방법...이 뭘까... 