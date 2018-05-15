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

public class myAgent_continue implements Runnable {

	private ActionRobot aRobot;
	private Random randomGenerator;
	public int currentLevel = 1;
	public static int time_limit = 12;
	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private int shotNumber = 1;
	private int sizeofbirds = 0;
	private double min_angle = 0.174533; //10도 //0;
	private double angle = min_angle; //(level 1성공하는 최소 각도, 대충 0.23 라디안) for test
	private double max_angle = 0.226893; //13도 Math.PI/2;
	
	private int max_pig = 9;
	
	// for CSV file
//	private static ArrayList<ArrayList<String>> info_set_level = new ArrayList<ArrayList<String>>();
//	private static ArrayList<ArrayList<String>> info_set_total = new ArrayList<ArrayList<String>>();
//	private static ArrayList<String> info_oneshot = new ArrayList<String>();
//	private static ArrayList<String> info_pigs_loc = new ArrayList<String>();
//	private static ArrayList<String> info_obs_loc = new ArrayList<String>();
	private ArrayList<String> info_field = new ArrayList<String>
	(Arrays.asList("Level", "ShotNum", "Angle", "TapTime", "BirdType", "ImageNmae", "Score", "State", "Pigs", "Obstacle"));
	
	// a stand-alone implementation of the Naive Agent
	public myAgent_continue() {
		
		aRobot = new ActionRobot();
		tp = new TrajectoryPlanner();
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
		
		ArrayList<ArrayList<String>> info_set_level = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<String>> info_set_total = new ArrayList<ArrayList<String>>();
		info_set_level.add(info_field); //CSV 출력을 위해 각 column name 저장
		info_set_total.add(info_field);
		
		aRobot.loadLevel(currentLevel);
		System.out.println("\n==========LEVEL " + currentLevel + "==========");
		
		String pwd = System.getProperty("user.dir"); // CSV 저장을 위한 현재 폴더 위치 얻기
		
		
		
		while (true) {
			
			ArrayList<String> info_oneshot = new ArrayList<String>(); // 새 객체 할당, 사용후 null으로 해제 
			ArrayList<String> info_pigs_loc = new ArrayList<String>(); // 새 객체 할당
			ArrayList<String> info_obs_loc = new ArrayList<String>(); // 새 객체 할당
			
			info_oneshot.add(String.valueOf(currentLevel)); 
			info_oneshot.add(String.valueOf(shotNumber)); 

			GameState state = solve(info_oneshot, info_pigs_loc, info_obs_loc); 					
			System.out.println("One shot solved; current state is "+state);
			
			if (state == GameState.WON||state == GameState.LOST||state == GameState.PLAYING) {
				info_oneshot.add(state.toString());
				info_oneshot.addAll(info_pigs_loc);
				info_oneshot.addAll(info_obs_loc);
				
				System.out.println("information_oneshot:");
				InfoCSV.print_info(info_oneshot);
				System.out.print("\n");
				
				info_set_level.add(info_oneshot);
				info_set_total.add(info_oneshot);
				
				// 쓰고 난 객체 해제... 우선 null로 처리... 
				info_oneshot = null; //new ArrayList<String>(); 
				info_pigs_loc = null; //new ArrayList<String>();
				info_obs_loc = null; //new ArrayList<String>();
			}
			if (state == GameState.WON) { 
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				
				int score = StateUtil.getScore(ActionRobot.proxy); // WON한 상태에서 accumulated score
				if(!scores.containsKey(currentLevel)) // scores에 최고 점수 갱신
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
				
				if (angle < max_angle) { //이겼을때도 계속해서 0.5도씩 증가시키며 실행
					aRobot.restartLevel();
					angle += Math.PI/360;

				} else { // 이기고 앵글테스트도 끝나면 다음 레벨로
					// CSV 출력
					String filepath_level = pwd+"/info_"+currentLevel+".csv"; 
					InfoCSV.writecsv(info_set_level, filepath_level);
					
					InfoCSV.print_infoset(info_set_level);
					System.out.println("CSV Save: " + filepath_level);
					
					// 여기도 새 객체... 할당... 컴터에게 못할짓... ㅠ.ㅠ 
					info_set_level = null;
					info_set_level = new ArrayList<ArrayList<String>>();
					info_set_level.add(info_field);
					
					if (currentLevel == 21) {
						String filepath_total = pwd+"/info.csv";
						InfoCSV.writecsv(info_set_total, filepath_total);
						return;
					}
					
					// Go to the next level
					aRobot.loadLevel(++currentLevel);
					System.out.println("\n==========LEVEL " + currentLevel + "==========");
					angle = min_angle;

					// make a new trajectory planner whenever a new level is entered
					tp = new TrajectoryPlanner();
				}
				shotNumber = 1;
				
			} else if (state == GameState.LOST) {
				
				if (angle < max_angle) { //졌을때도 계속해서 0.5도씩 증가시키며 실행
					System.out.println("Lost and Restart");
					aRobot.restartLevel();
					angle += Math.PI/360;
				} else { // 졌지만, 이 레벨에서 정한 각도로 다 쏴봤으면 CSV로 파일 저장
					// CSV 출력
					String filepath_level = pwd+"/info_"+currentLevel+".csv"; 
					InfoCSV.writecsv(info_set_level, filepath_level);
					
					InfoCSV.print_infoset(info_set_level);
					System.out.println("CSV Save: " + filepath_level);
					
					// 여기도 새 객체... 할당... 컴터에게 못할짓... ㅠ.ㅠ 
					info_set_level = null;
					info_set_level = new ArrayList<ArrayList<String>>();
					info_set_level.add(info_field);
					
					if (currentLevel == 21) {
						String filepath_total = pwd+"/info.csv";
						InfoCSV.writecsv(info_set_total, filepath_total);
						return;
					}
					
					// Go to the next level
					aRobot.loadLevel(++currentLevel);
					System.out.println("\n==========LEVEL " + currentLevel + "==========");
					angle = min_angle;
					
					// make a new trajectory planner whenever a new level is entered
					tp = new TrajectoryPlanner();
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
			}

		} // while(true) 괄호
	} // public run() 괄호
	
	public GameState solve(ArrayList<String> info_oneshot, ArrayList<String> info_pigs_loc, ArrayList<String> info_obs_loc)
	{
		// Remove side menu for sizeofbirds/ screenshot
		ActionRobot.RemoveSideMenu(); // 화면 중앙을 클릭해서 sidemenu를 없앰
		
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
 		
 		// get obstacle location and type
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
				
				if(shotNumber==1) {
					List<ABObject> birds = vision.findBirdsMBR(); // solve에 들어오자마자 캡쳐한 screenshot
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
				if (releasePoint != null) {					ABType type = aRobot.getBirdTypeOnSling(); //unknown인 애가 뜰때도 있음.... ㅜ.ㅜ 뭐지
					info_oneshot.add(String.valueOf(taptime));
					info_oneshot.add(type.toString());
					// tap-time 1000으로 줬는데, 새마다 학습해야 함
						
					dx = (int)releasePoint.getX() - refPoint.x;
					dy = (int)releasePoint.getY() - refPoint.y;
					shot = new Shot(refPoint.x, refPoint.y, dx, dy, taptime);
				} else {
					System.err.println("No Release Point Found");
					return state;
				}
			} // pig가 empty가 아닐때 괄호
			

			// check whether the slingshot is changed. 
			// the change of the slingshot indicates a change in the scale.
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
								screenshot = ActionRobot.doScreenShot(); // shoot을 하고, 다시 screenshot을 봄
								vision = new Vision(screenshot); 
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, releasePoint);							
							}
						} 
					}
					else
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
				}
				else
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");
			}
			int score =0;
			
			System.out.println("Shot angle: "+Math.toDegrees(angle)+
					" || sizeofbirds: "+ sizeofbirds+" || shotNumber: " +(shotNumber-1));
			
			// 돼지가 없는지 확인 (돼지가 없으면 WON을 기다리면 됨)
			pigs = vision.findPigsMBR(); // state에서 shoot을 하고나면 pig의 상태는 바로 반영이 되나??  
			if(pigs.isEmpty()) { 
				while(true) {
					if (aRobot.getState()==GameState.WON) {
						System.out.println("pigs.isEmpty...state gets won!");
						score = StateUtil.getScore(ActionRobot.proxy)- 10000*(sizeofbirds-shotNumber+1); //shoot score
//						info_oneshot.add(String.valueOf(StateUtil.getScore(ActionRobot.proxy))); // accumulated score
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
			
			if(sizeofbirds==shotNumber-1) { // 그 level에서 shoot을 다함
				while(true) {
					int temp = StateUtil.getScore(ActionRobot.proxy);
					if (temp>score) { // 0보다 큰 마지막 점수를 받아오도록
						score = temp;
					}
					if (aRobot.getState()==GameState.LOST) {
						System.out.println("birds.isEmpty...state gets lost!");
//						info_oneshot.add(String.valueOf(StateUtil.getScore(ActionRobot.proxy)));
						System.out.println("lost score" + score);
						info_oneshot.add(String.valueOf(score));
						return aRobot.getState();
					}
				}
			}
			
			// 돼지는 있고 새도 있을 때, 안정화된 점수를 기다림
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

		return state;
	}

	public static void main(String args[]) {

		myAgent_continue na = new myAgent_continue();
		if (args.length > 0)
			na.currentLevel = Integer.parseInt(args[0]);
		na.run();

	}
}
