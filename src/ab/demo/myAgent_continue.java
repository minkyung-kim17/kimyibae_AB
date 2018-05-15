/*****************************************************************************
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014, XiaoYu (Gary) Ge, Stephen Gould, Jochen Renz
 **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
 ** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/
package ab.demo;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

import ab.demo.other.ActionRobot;
import ab.demo.other.Shot;
import ab.planner.TrajectoryPlanner;
import ab.utils.StateUtil;
import ab.vision.ABObject;
import ab.vision.ABType;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.Vision;

import ab.utils.InfoCSV;

// 변경 
public class myAgent_continue implements Runnable {

	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
	public static int time_limit = 12;
	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private int shotNumber = 1;
	private int sizeofbirds = 0;
	private Point prevTarget;
	private double min_angle = 0;
	private double angle = min_angle; //(level 1성공하는 최소 각도, 대충 0.23 라디안) //0;
	private double max_angle = Math.PI/36; //Math.PI/2; 
	
	private int max_pig = 9;
	// pi/3이었을때, 1.0471975511965983 이 각도에서 못넘어가고 계속 걸림... 
	// 밑에 game state들을 다 고려안해서 그런것임.. 
	// 그래서 저장을 못함..ㅠ.ㅠ
	
	// for CSV file
	private static ArrayList<ArrayList<String>> info_set_level = new ArrayList<ArrayList<String>>();
	private static ArrayList<ArrayList<String>> info_set_total = new ArrayList<ArrayList<String>>();
	private static ArrayList<String> info_oneshot = new ArrayList<String>();
	private static ArrayList<String> info_pigs_loc = new ArrayList<String>();
	private static ArrayList<String> info_obs_loc = new ArrayList<String>();
	private ArrayList<String> info_field = new ArrayList<String>(Arrays.asList("Level", "ShotNum", "Angle", "TapTime", "BirdType", "ImageNmae", "Score", "State", "Pigs", "Obstacle"));
	
	// a standalone implementation of the Naive Agent
	public myAgent_continue() {
		
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
		prevTarget = null;
		shotNumber= 1;
		randomGenerator = new Random();
		// --- go to the Poached Eggs episode level selection page ---
		ActionRobot.GoFromMainMenuToLevelSelection();

	}

	// get current time for debug
	public static String getCurrentTime(String timeFormat) {
		return new SimpleDateFormat(timeFormat).format(System.currentTimeMillis());
	}
	
	// run the client
	public void run() {

		aRobot.loadLevel(currentLevel);
		System.out.println("\n==========LEVEL " + currentLevel + "==========");
		
		String pwd = System.getProperty("user.dir"); // CSV 저장을 위해 현재 폴더 위치 얻기
		
		info_set_level.add(info_field);
		info_set_total.add(info_field);
		
		while (true) {
			
			info_oneshot.add(String.valueOf(currentLevel)); // add_info_level
			info_oneshot.add(String.valueOf(shotNumber));

			GameState state = solve(); // add_info_angle, add_info_score, add_info_pigsLocation
//			System.out.println(state);
					
			System.out.println("solved "+state);
			if (state == GameState.WON||state == GameState.LOST||state == GameState.PLAYING) {
				info_oneshot.add(state.toString());
//				info_oneshot.set(2, String.valueOf(totalScore)); // 최종 점수로 갱신, index 위치 "score"위치에 따라 바뀔수 있도록 수정필요
				// 여기가... 
				info_oneshot.addAll(info_pigs_loc);
				info_oneshot.addAll(info_obs_loc);
				System.out.println("information_oneshot:");
				InfoCSV.print_info(info_oneshot);
				System.out.print("\n");
				
				info_set_level.add(info_oneshot);
				info_set_total.add(info_oneshot);
				
				info_oneshot = new ArrayList<String>(); // 다음 shot information 저장을 위해 새 객체 할당
				info_pigs_loc = new ArrayList<String>();
				info_obs_loc = new ArrayList<String>();
			}
			if (state == GameState.WON) { //
//				String nowTime2 = getCurrentTime("mm:ss");
//				System.out.println(nowTime2);
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				int score = StateUtil.getScore(ActionRobot.proxy);
				if(!scores.containsKey(currentLevel))
					scores.put(currentLevel, score);
				else
				{
					if(scores.get(currentLevel) < score)
						scores.put(currentLevel, score);
				}
				int totalScore = 0;
				for(Integer key: scores.keySet()){

					totalScore += scores.get(key);
//					System.out.println(" Level " + key
//							+ " Score: " + scores.get(key) + " ");
				}
//				System.out.println("Total Score: " + totalScore);
				
				/////////////////////////////////////////////이겼을 때 결과 저장?

				/////////////////////////////////////////////
				
				if (angle < max_angle) { //이겼을때도 일단 여기 다시 시작
					aRobot.restartLevel();
					angle += Math.PI/360;

				} else { //이기고 앵글테스트도 끝나면 다음 레벨로
					// 정보 저장
					String filepath_level = pwd+"/info_"+currentLevel+".csv"; 
					InfoCSV.writecsv(info_set_level, filepath_level);
					InfoCSV.print_infoset(info_set_level);
					info_set_level = new ArrayList<ArrayList<String>>();
					info_set_level.add(info_field);
					System.out.println(filepath_level);
					
					if (currentLevel == 21) {
						String filepath_total = pwd+"/info.csv";
						InfoCSV.writecsv(info_set_total, filepath_total);
						return;
					}
					
					// 다음 level 진행
					aRobot.loadLevel(++currentLevel);
					System.out.println("\n==========LEVEL " + currentLevel + "==========");
					angle = min_angle;

					// make a new trajectory planner whenever a new level is entered
					tp = new TrajectoryPlanner();

					// first shot on this level, try high shot first
//					firstShot = true;
				}
				shotNumber = 1;
			} else if (state == GameState.LOST) {


				if (angle < max_angle) { //이겼을때도 일단 여기 다시 시작
					System.out.println("Restart");
					aRobot.restartLevel();
					angle += Math.PI/360;
				} else { //이기고 앵글테스트도 끝나면 다음 레벨로
					// 정보 저장
					String filepath_level = pwd+"/info_"+currentLevel+".csv"; 
					InfoCSV.writecsv(info_set_level, filepath_level);
					InfoCSV.print_infoset(info_set_level);
					info_set_level = new ArrayList<ArrayList<String>>();
					info_set_level.add(info_field);
					System.out.println(filepath_level);
					
					if (currentLevel == 21) {
						String filepath_total = pwd+"/info.csv";
						InfoCSV.writecsv(info_set_total, filepath_total);
						return;
					}
					
					// 다음 level 진행
					aRobot.loadLevel(++currentLevel);
					System.out.println("\n==========LEVEL " + currentLevel + "==========");
					angle = min_angle;
					
					// make a new trajectory planner whenever a new level is entered
					tp = new TrajectoryPlanner();

					// first shot on this level, try high shot first
//					firstShot = true;
				}
				shotNumber = 1;

				
			} else if (state == GameState.LEVEL_SELECTION) {
				System.out
				.println("Unexpected level selection page, go to the last current level : "
						+ currentLevel);
				aRobot.loadLevel(currentLevel);
			} else if (state == GameState.MAIN_MENU) {
				System.out
				.println("Unexpected main menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				aRobot.loadLevel(currentLevel);
			} else if (state == GameState.EPISODE_MENU) {
				System.out
				.println("Unexpected episode menu page, go to the last current level : "
						+ currentLevel);
				ActionRobot.GoFromMainMenuToLevelSelection();
				aRobot.loadLevel(currentLevel);
			} else if(state == GameState.PLAYING) {
				
				// make a new trajectory planner whenever a new level is entered
				tp = new TrajectoryPlanner();

				// first shot on this level, try high shot first
//				firstShot = false;
				
			}

		}

	}

	public GameState solve()
	{

		// capture Image
		BufferedImage screenshot = ActionRobot.doScreenShot();

		// process image
		Vision vision = new Vision(screenshot);
				
		// find the slingshot
		Rectangle sling = vision.findSlingshotMBR();
		
		// confirm the slingshot
		while (sling == null && aRobot.getState() == GameState.PLAYING) {
			System.out
			.println("No slingshot detected. Please remove pop up or zoom out");
			ActionRobot.fullyZoomOut();
			screenshot = ActionRobot.doScreenShot();
			vision = new Vision(screenshot);
			sling = vision.findSlingshotMBR();
		}
		
		Point releasePoint = null;
		Shot shot = new Shot();
		int dx = 0;
		int dy = 0;

		// get all the pigs
 		List<ABObject> pigs = vision.findPigsMBR();
 		for (ABObject pig : pigs) { 
 			info_pigs_loc.add(Double.toString(pig.getCenterX()));
 			info_pigs_loc.add(Double.toString(pig.getCenterY()));
 		} // get pig location
 		for (int i=0; i<max_pig-pigs.size(); i++) {
 			info_pigs_loc.add("0");
 			info_pigs_loc.add("0");
 		}
 		
 		List<ABObject> obstacles = vision.findBlocksMBR();
 		for (ABObject obstacle : obstacles) {
 			info_obs_loc.add(obstacle.type.toString());
 			info_obs_loc.add(Double.toString(obstacle.getCenterX()));
 			info_obs_loc.add(Double.toString(obstacle.getCenterY()));
 		}
 		
		GameState state = aRobot.getState();
		int taptime = 1000;
		String sspwd = System.getProperty("user.dir");
		String currentTime = getCurrentTime("hhmmss");
		// if there is a sling, then play, otherwise just skip.
		if (sling != null) {

			if (!pigs.isEmpty()) {
				//first shot이 아닐 경우
				if(shotNumber==1) {
					List<ABObject> birds = vision.findBirdsMBR();
					sizeofbirds = birds.size();
					releasePoint = tp.findReleasePoint(sling, angle);
					info_oneshot.add(Double.toString(angle));
				}
				if(shotNumber>1) {
					//gaussian분포 std를 PI/8
					double twice_stdval = Math.PI/8;
					//현재 쏜 angle을 mean으로
					double mean = angle;
					double tempangle = mean+(randomGenerator.nextGaussian())*twice_stdval;
					//min angle, max angle 넘어가면 mean으로 set.
					if (tempangle<0 || tempangle>Math.PI) {
						tempangle = mean;
					}
					releasePoint = tp.findReleasePoint(sling, tempangle);
					info_oneshot.add(Double.toString(tempangle));

				}

				// Get the reference point
				Point refPoint = tp.getReferencePoint(sling);

				//Calculate the tapping time according the bird type 
				if (releasePoint != null) {					ABType type = aRobot.getBirdTypeOnSling(); 
					info_oneshot.add(String.valueOf(taptime));
					info_oneshot.add(type.toString());
					// taptime 1000으로 줬는데, 새마다 학습해야 함
						
					dx = (int)releasePoint.getX() - refPoint.x;
					dy = (int)releasePoint.getY() - refPoint.y;
					shot = new Shot(refPoint.x, refPoint.y, dx, dy, taptime);
				} else {
					System.err.println("No Release Point Found");
					return state;
				}
			}
			

			// check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
			{
				ActionRobot.fullyZoomOut();
				screenshot = ActionRobot.doScreenShot();
				try {
				    // retrieve image
				    File outputfile = new File(sspwd+"/screenshot/screenshot_level"+currentLevel+"_"+currentTime+".png");
				    
				    ImageIO.write(screenshot, "png", outputfile);
				    info_oneshot.add(outputfile.getName());					
				} catch (IOException e) {
					e.printStackTrace();
				}
				vision = new Vision(screenshot);
				Rectangle _sling = vision.findSlingshotMBR();
				if(_sling != null)
				{
					double scale_diff = Math.pow((sling.width - _sling.width),2) +  Math.pow((sling.height - _sling.height),2);	
					if(scale_diff < 25)
					{
						if(dx < 0)
						{
//							try {
//							    // retrieve image
//							    File outputfile = new File(sspwd+"/screenshot/screenshot_level"+currentLevel+"_"+currentTime+".png");
//							    
//							    ImageIO.write(screenshot, "png", outputfile);
//							    info_oneshot.add(outputfile.getName());					
//							} catch (IOException e) {
//								e.printStackTrace();
//							}
							aRobot.cshoot(shot);
							shotNumber ++;
							state = aRobot.getState();								
							if ( state == GameState.PLAYING )
							{
								screenshot = ActionRobot.doScreenShot();
								vision = new Vision(screenshot);
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, releasePoint);							
							}
						} //else { // GameState가 playing이 아니어도 (angle, score)에 nan 입력
//							info_oneshot.add("nan");
//							info_oneshot.add("nan");
//						}
					}
					else
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
				}
				else
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");
			}
			int score =0;
			// 돼지가 없는지 확인 (돼지가 없으면 WON을 기다리면 됨)
			pigs = vision.findPigsMBR();
			if(pigs.isEmpty()) { 
				while(true) {
					if (aRobot.getState()==GameState.WON) {
						System.out.println("pigs.isEmpty...state gets won!");
						score = StateUtil.getScore(ActionRobot.proxy)- 10000*(sizeofbirds-shotNumber+1);
//						info_oneshot.add(String.valueOf(StateUtil.getScore(ActionRobot.proxy)));
						info_oneshot.add(String.valueOf(score)); 
						return aRobot.getState();
					}
				}
			}	
			// 돼지는 있고, 새가 없는지 확인 (새가 없으면 lost를 기다리면 됨)
//			List<ABObject> birds = vision.findBirdsMBR();
//			if(birds.isEmpty()) { 
//				while(true) {
//					if (aRobot.getState()==GameState.LOST) {
//						System.out.println("birds.isEmpty...state gets lost!");
//						return state;
//					}
//				}
//			}
			System.out.println("sizeofbirds: "+ sizeofbirds+" || shotNumber: " +(shotNumber-1));
			if(sizeofbirds==shotNumber-1) { 
				while(true) {
					int temp = StateUtil.getScore(ActionRobot.proxy);
					if (temp>=score) {
						score = temp;
						System.out.println("in for " + score);
					}
					if (aRobot.getState()==GameState.LOST) {
						System.out.println("birds.isEmpty...state gets lost!");
//						info_oneshot.add(String.valueOf(StateUtil.getScore(ActionRobot.proxy)));
						System.out.println("out for " + score);
						info_oneshot.add(String.valueOf(score));
						return aRobot.getState();
					}
				}
			}
			
			//돼지는 있고 새도 있을 때, 기다림
			try {
				System.out.println("sleeping...");
				Thread.sleep(10000);
				System.out.println("wake up...");
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
//			int score = StateUtil.getScore(ActionRobot.proxy); 
			info_oneshot.add(String.valueOf(StateUtil.getScore(ActionRobot.proxy))); 
		} 

			
//		System.out.println("solved, state: " + state.toString());		
		return state;
	}

	public static void main(String args[]) {

		myAgent_continue na = new myAgent_continue();
		if (args.length > 0)
			na.currentLevel = Integer.parseInt(args[0]);
		na.run();

	}
}
