package ab.demo;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import ab.demo.other.ActionRobot;
import ab.demo.other.ClientActionRobot;
import ab.demo.other.ClientActionRobotJava;
import ab.demo.other.Shot;
import ab.planner.TrajectoryPlanner;
import ab.utils.StateUtil;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;

import ab.utils.InfoCSV;

//Naive agent (server/client version)

public class myClientAgent implements Runnable { ///에러있음 ... 

	//Wrapper of the communicating messages
	private ClientActionRobotJava ar;
	public byte currentLevel = -1;
	public int failedCounter = 0;
	public int[] solved;
	TrajectoryPlanner tp; 
	private int id = 28888;
	private boolean firstShot;
	private Point prevTarget;
	private Random randomGenerator;
	private double angle = 0; //(level 1성공하는 최소 각도, 대충 0.23 라디안) //0;
	private double max_angle = Math.PI/2; //90-degree
	
	private static ArrayList<ArrayList<String>> info_set_level = new ArrayList<ArrayList<String>>();
	private static ArrayList<ArrayList<String>> info_set_total = new ArrayList<ArrayList<String>>();
	private static ArrayList<String> info_oneshot = new ArrayList<String>();
	private static ArrayList<String> info_pigs_loc = new ArrayList<String>();
	private ArrayList<String> info_field = new ArrayList<String>(Arrays.asList("Level", "Angle", "Score", "State", "Pigs"));
	
	/**
	 * Constructor using the default IP
	 * */
	public myClientAgent() {
		// the default ip is the localhost
		ar = new ClientActionRobotJava("127.0.0.1");
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;

	}
	/**
	 * Constructor with a specified IP
	 * */
	public myClientAgent(String ip) {
		ar = new ClientActionRobotJava(ip);
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;

	}
	public myClientAgent(String ip, int id)
	{
		ar = new ClientActionRobotJava(ip);
		tp = new TrajectoryPlanner();
		randomGenerator = new Random();
		prevTarget = null;
		firstShot = true;
		this.id = id;
	}
	public int getNextLevel()
	{
		int level = 0;
		boolean unsolved = false;
		//all the level have been solved, then get the first unsolved level
		for (int i = 0; i < solved.length; i++)
		{
			if(solved[i] == 0 )
			{
					unsolved = true;
					level = i + 1;
					if(level <= currentLevel && currentLevel < solved.length)
						continue;
					else
						return level;
			}
		}
		if(unsolved)
			return level;
	    level = (currentLevel + 1)%solved.length;
		if(level == 0)
			level = solved.length;
		return level; 
	}
    /* 
     * Run the Client (Naive Agent)
     */
	private void checkMyScore()
	{
		
		int[] scores = ar.checkMyScore();
		System.out.println(" My score: "); //가장 먼저 나오는 print 
		int level = 1;
		for(int i: scores)
		{
			System.out.println(" level " + level + "  " + i);
			if (i > 0)
				solved[level - 1] = 1;
			level ++;
		}
	}
	
	public void run() {
		
		byte[] info = ar.configure(ClientActionRobot.intToByteArray(id));
		solved = new int[info[2]];
		
		//load the initial level (default 1)
		//Check my score
		checkMyScore();
		
		currentLevel = (byte)getNextLevel();
		ar.loadLevel(currentLevel);
		System.out.println("\n==========LEVEL " + currentLevel + "=========="); //

		GameState state;
		
		String pwd = System.getProperty("user.dir");
		
		info_set_level.add(info_field);
		info_set_total.add(info_field);
		
		while (true) {
			
			info_oneshot.add(String.valueOf(currentLevel)); // 되는거 맞는지? 
			System.out.println("here");
			state = solve();
			
			//If the level is solved , go to the next level
			if (state == GameState.WON) {
				
				///System.out.println(" loading the level " + (currentLevel + 1) );
				checkMyScore();
				System.out.println();
				currentLevel = (byte)getNextLevel(); 
				ar.loadLevel(currentLevel);
				//ar.loadLevel((byte)9);
				//display the global best scores
				int[] scores = ar.checkScore();
				System.out.println("Global best score: ");
				for (int i = 0; i < scores.length ; i ++)
				{
				
					System.out.print( " level " + (i+1) + ": " + scores[i]);
				}
				System.out.println();
				
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
				firstShot = true;
				
			} else 
				//If lost, then restart the level
				if (state == GameState.LOST) {
				failedCounter++;
				if(failedCounter > 3)
				{
					failedCounter = 0;
					currentLevel = (byte)getNextLevel(); 
					ar.loadLevel(currentLevel);
					
					//ar.loadLevel((byte)9);
				}
				else
				{		
					System.out.println("restart");
					ar.restartLevel();
				}
						
			} else 
				if (state == GameState.LEVEL_SELECTION) {
				System.out.println("unexpected level selection page, go to the last current level : "
								+ currentLevel);
				ar.loadLevel(currentLevel);
			} else if (state == GameState.MAIN_MENU) {
				System.out
						.println("unexpected main menu page, reload the level : "
								+ currentLevel);
				ar.loadLevel(currentLevel);
			} else if (state == GameState.EPISODE_MENU) {
				System.out.println("unexpected episode menu page, reload the level: "
								+ currentLevel);
				ar.loadLevel(currentLevel);
			}

		}

	}


	  /** 
	   * Solve a particular level by shooting birds directly to pigs
	   * @return GameState: the game state after shots.
     */
	public GameState solve()

	{
		System.out.println("here");
		// capture Image
		BufferedImage screenshot = ar.doScreenShot();
		System.out.println("here");
		// process image
		Vision vision = new Vision(screenshot);
		System.out.println("here2");
		Rectangle sling = vision.findSlingshotMBR();
		System.out.println("here3");
		//If the level is loaded (in PLAYING state)but no slingshot detected, then the agent will request to fully zoom out.
		while (sling == null && ar.checkState() == GameState.PLAYING) {
			System.out.println("no slingshot detected. Please remove pop up or zoom out");
			
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				
				e.printStackTrace();
			}
			ar.fullyZoomOut();
			screenshot = ar.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotMBR();
		}

		Point releasePoint = null;
		Shot shot = new Shot();
		int dx = 0;
		int dy = 0;
		
		 // get all the pigs
 		List<ABObject> pigs = vision.findPigsMBR();
 		for (ABObject pig : pigs) { // ABObject pig = pigs.get(pigs.size()-1);
 			info_pigs_loc.add(Double.toString(pig.getCenterX()));
 			info_pigs_loc.add(Double.toString(pig.getCenterY()));
 		} // get pig location
 		System.out.println("here4"); //ok
		
 		GameState state = ar.checkState();
		int taptime = 1000;
		
		// if there is a sling, then play, otherwise skip.
		if (sling != null) {
			System.out.println("here5");
			//If there are pigs, we pick up a pig randomly and shoot it. 
			if (!pigs.isEmpty()) {		
				System.out.println("here5");
				releasePoint = tp.findReleasePoint(sling, angle);
				if (angle<=max_angle) { // 60-degree
					angle = angle+Math.PI/360; // 0.5-degree
				}
				info_oneshot.add(Double.toString(angle));
//				info_oneshot.add(Double.toString(Math.toDegrees(angle)));

				// 다음으로, 여기에서는 우선 first shot만 보는 거라, random pig selection 안함
				
				// Get the reference point
				Point refPoint = tp.getReferencePoint(sling);
					
				int tapTime = 0;
				if (releasePoint != null) {
//					ABType type = ar.getBirdTypeOnSling();
					// 새 종류별로 taptime 학습필요

					dx = (int)releasePoint.getX() - refPoint.x;
					dy = (int)releasePoint.getY() - refPoint.y;
					shot = new Shot(refPoint.x, refPoint.y, dx, dy, taptime);
				} else{
					System.err.println("No Release Point Found");
					return ar.checkState();
				}
				
				// check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
				ar.fullyZoomOut();
				screenshot = ar.doScreenShot();
				vision = new Vision(screenshot);
				Rectangle _sling = vision.findSlingshotMBR();
				if(_sling != null)
				{
					double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);
					if(scale_diff < 25)
					{
						if(dx < 0)
						{
							long timer = System.currentTimeMillis();
							ar.shoot(refPoint.x, refPoint.y, dx, dy, 0, tapTime, false);
//							System.out.println("It takes " + (System.currentTimeMillis() - timer) + " ms to take a shot");
							state = ar.checkState();
							if ( state == GameState.PLAYING )
							{
								screenshot = ar.doScreenShot();
								vision = new Vision(screenshot);
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, releasePoint);
								firstShot = false;
							}
						} else { // GameState가 playing이 아니어도 (angle, score)에 nan 입력
							info_oneshot.add("nan");
							info_oneshot.add("nan");
						}
					}
					else
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
				}
				else
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");
				
			}
			int score = StateUtil.getScore(ActionRobot.proxy);
			info_oneshot.add(String.valueOf(score));
//			System.out.println("during the game state... score : " + score); // score는 shot 점수? 
			} else { //sling을 못찾은 경우, solve 끝내고 game state 확인으로 돌아가게 됨. 
				System.out.println("hereF");
				info_pigs_loc.add("SlingEmpty");
				info_oneshot.add("nan");
				info_oneshot.add("nan");
		}
		System.out.println("solved, state: " + state.toString());
	
		return state;
	}

	private double distance(Point p1, Point p2) {
		return Math.sqrt((double) ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)* (p1.y - p2.y)));
	}

	public static void main(String args[]) {

		myClientAgent na;
		if(args.length > 0)
			na = new myClientAgent(args[0]);
		else
			na = new myClientAgent();
		
		na.run();
		
	}
}
