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
//	private Map<Integer,Integer> scores = new LinkedHashMap<Integer,Integer>();
	TrajectoryPlanner tp;
	private int shotNumber = 1;
	private double min_angle = Math.PI/2-Math.PI/180; //10�� //0;
	private double angle = min_angle; //(level 1�����ϴ� �ּ� ����, ���� 0.23 ������) for test
	private double max_angle = Math.PI/2-Math.PI/360; //13�� Math.PI/2;
	private int max_pig = 9;
	private int last_score = 0;


	private ArrayList<String> info_field = new ArrayList<String>
	(Arrays.asList("Level", "ShotNum", "Angle", "TapTime", "BirdType", "ImageNmae", "Score-lastScore", "Score", "State", "Pigs", "Obstacle"));

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
		info_set_level.add(info_field); //CSV ������ ���� �� column name ����
		info_set_total.add(info_field);

		aRobot.loadLevel(currentLevel);
		System.out.println("\n==========LEVEL " + currentLevel + "==========");

		String pwd = System.getProperty("user.dir"); // CSV ������ ���� ���� ���� ��ġ ����



		while (true) {

			ArrayList<String> info_oneshot = new ArrayList<String>(); // �� ��ü �Ҵ�, ������ null���� ����
			ArrayList<String> info_pigs_loc = new ArrayList<String>(); // �� ��ü �Ҵ�
			ArrayList<String> info_obs_loc = new ArrayList<String>(); // �� ��ü �Ҵ�

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

				// ���� �� ��ü ����... �켱 null�� ó��...
				info_oneshot = null; //new ArrayList<String>();
				info_pigs_loc = null; //new ArrayList<String>();
				info_obs_loc = null; //new ArrayList<String>();

				if(state == GameState.WON||state == GameState.LOST) {
					if (angle >= max_angle-Math.PI/720) {
						// CSV ����
						String currentTime = getCurrentTime("MMDDHHmmss");
						String filepath_level = pwd+"/infolevel_"+currentLevel+"_"+currentTime+".csv";
						InfoCSV.writecsv(info_set_level, filepath_level);

						InfoCSV.print_infoset(info_set_level);
						System.out.println("CSV Save: " + filepath_level);

						// ���⵵ �� ��ü... �Ҵ�... ���Ϳ��� ������... ��.��
						info_set_level = null;
						info_set_level = new ArrayList<ArrayList<String>>();
						info_set_level.add(info_field);

						if (currentLevel == 21) {
							String filepath_total = pwd+currentTime+"/info.csv";
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
					
					else { //�̰������� �����ؼ� 0.5���� ������Ű�� ����
						aRobot.restartLevel();
						angle += Math.PI/360;

					} 
					shotNumber = 1;
				}
			}
			if (state == GameState.WON) {
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}

			} else if (state == GameState.LOST) {

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

			}

		} // while(true) ��ȣ
	} // public run() ��ȣ

	public GameState solve(ArrayList<String> info_oneshot, ArrayList<String> info_pigs_loc, ArrayList<String> info_obs_loc)
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
			System.out.flush();
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
		String currentTime = getCurrentTime("MMDDHHmmss");

		// if there is a sling, then play, otherwise just skip.
		if (sling != null) {
			if(angle >= max_angle) {
				return GameState.LOST;
			}
			
			if (!pigs.isEmpty()) {

				if(shotNumber==1) {
					releasePoint = tp.findReleasePoint(sling, angle);
					info_oneshot.add(Double.toString(angle));
					last_score = 0;
				}
				if(shotNumber>1) {
					//gaussian���� std�� PI/8
					double twice_stdval = Math.PI/8;
					//���� �� angle�� mean����
					double mean = angle;
					double tempangle = mean+(randomGenerator.nextGaussian())*twice_stdval;
					//min angle, max angle �Ѿ�� mean���� set.
					if (tempangle<0 || tempangle>Math.PI) {
						tempangle = mean;
					}
					releasePoint = tp.findReleasePoint(sling, tempangle);
					info_oneshot.add(Double.toString(tempangle));
				}

				// Get the reference point
				Point refPoint = tp.getReferencePoint(sling);

				//Calculate the tapping time according the bird type
				if (releasePoint != null) {
					ABType type = aRobot.getBirdTypeOnSling(); //unknown�� �ְ� �㶧�� ����.... ��.�� ����
					info_oneshot.add(String.valueOf(taptime));
					info_oneshot.add(type.toString());
					// tap-time 1000���� ���µ�, ������ �н��ؾ� ��

					dx = (int)releasePoint.getX() - refPoint.x;
					dy = (int)releasePoint.getY() - refPoint.y;
					shot = new Shot(refPoint.x, refPoint.y, dx, dy, taptime);
				} else {
					System.err.println("No Release Point Found");
					return state;
				}
			} // pig�� empty�� �ƴҶ� ��ȣ


			// check whether the slingshot is changed.
			// the change of the slingshot indicates a change in the scale.

			{
				ActionRobot.fullyZoomOut();
				screenshot = ActionRobot.doScreenShot();
				try {
				    // retrieve image
				    File outputfile = new File(sspwd+"/screenshot/screenshot_level"+currentLevel+"_"+currentTime+"PLAYING.png");

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
								screenshot = ActionRobot.doScreenShot(); // shoot�� �ϰ�, �ٽ� screenshot�� ��
								vision = new Vision(screenshot);
								List<Point> traj = vision.findTrajPoints();
								tp.adjustTrajectory(traj, sling, releasePoint);
							}
							int score = StateUtil.getScore(ActionRobot.proxy);
							int sleepcount =0;
							System.out.println(sleepcount);
							while(true) {
								int temp = StateUtil.getScore(ActionRobot.proxy);
								if (temp>score) { // 0���� ū ������ ������ �޾ƿ�����
									score = temp;
									sleepcount = 0;
								}else if (temp==score) {
									if (aRobot.getState()==GameState.LOST||aRobot.getState()==GameState.WON) {
										break;
									}
									try {
										Thread.sleep(1000);
									} catch (InterruptedException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									}
									sleepcount+=1;
									if (sleepcount>2) {
										break;
									}
								}else if(temp<score){
									break;
								}
								System.out.println(sleepcount);
							}
							System.out.flush();
							info_oneshot.add(String.valueOf(score-last_score));
							info_oneshot.add(String.valueOf(score));
							last_score = score;
							

							if (aRobot.getState()==GameState.LOST) {
								screenshot = ActionRobot.doScreenShot();
								try {
										// retrieve image
										File outputfile = new File(sspwd+"/screenshot/screenshot_level"+currentLevel+"_"+currentTime+"LOST.png");
										ImageIO.write(screenshot, "png", outputfile);
										info_oneshot.add(outputfile.getName());					
								} catch (IOException e) {
									e.printStackTrace();
								}
//								return aRobot.getState();
							}
							if (aRobot.getState()==GameState.WON) {
								screenshot = ActionRobot.doScreenShot();
								try {
										// retrieve image
										File outputfile = new File(sspwd+"/screenshot/screenshot_level"+currentLevel+"_"+currentTime+"WON.png");
										ImageIO.write(screenshot, "png", outputfile);
										info_oneshot.add(outputfile.getName());					
								} catch (IOException e) {
									e.printStackTrace();
								}
//										return aRobot.getState();
							}
							
						}
					}
					else {
						System.out.println("Scale is changed, can not execute the shot, will re-segement the image");
						System.out.flush();
					}

				}
				else {
					System.out.println("no sling detected, can not execute the shot, will re-segement the image");
					System.out.flush();
				}

			}

			System.out.println("Shot angle: "+Math.toDegrees(angle)+" || shotNumber: " +(shotNumber-1));
			System.out.flush();

		}
		return aRobot.getState();
	}
//}

	public static void main(String args[]) {

		myAgent_continue na = new myAgent_continue();
		if (args.length > 0)
			na.currentLevel = Integer.parseInt(args[0]);
		na.run();

	}
}
