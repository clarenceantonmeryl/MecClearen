
# This project requires a good understanding of NUMPY library.

# Please refer to the following tutorial if needed at: 
# https://numpy.org/devdocs/user/quickstart.html

# Also a good understanding of Tensors in NUMPY is required.
# https://machinelearningmastery.com/introduction-to-tensors-for-machine-learning/

# To learn "ObjectProperty", "NumericProperty" and "ReferenceListProperty", 
# see kivy tutorials: https://kivy.org/docs/tutorials/pong.html

# Import numpy and plot.
import numpy as np
import matplotlib.pyplot as plot

# Import Kivy packages.
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config

# Spyder IDE may flag that kivy.graphics is unused but it is required for the robot.kv file.  
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.button import Button 
from kivy.vector import Vector
from kivy.core.window import Window
from kivy.uix.label import Label

from kivy.uix.slider import Slider

# Import the Deep_Q_Network object from mecclearen_brain.py.
from mecclearen_brain import Deep_Q_Network

# To remove the red dot when the mouse right click is pressed.
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# The last_x and last_y variables for the last points to be in memory 
# when I draw obstacles on the ocean.
last_x = 0
last_y = 0

# Margin offset.
margin_offset = 15


# This is the total number of points in the drawing.
n_points = 0

# The length of the drawing.
length = 0

# Initialising the last distance from the goal.
# The robot is rewarded based on how far or close they navigate during the training.
last_distance = 0

# Last reward given to the robot.
last_reward = 0

# Initial velocity
current_velocity = 9

# Initialise the AI, with "mecclearen_brain.py", which contains the neural network.
# 5 'dimensions' of input (3 input signals, positive and negative orientiation of the robot), 
# 3 actions or output (move forward, turn left, turn right), 
# Discount factor = 0.9 (gamma - in the Bellman equation)
# A discount of 0.9 is my choice, but you can choose your own value.
ai_brain = Deep_Q_Network(5, 3, 0.9) 

# 0 degrees = go straight, +21 degrees = turn left, -21 degrees = turn right.
# +21 degrees and -21 degrees are my choice, but you can choose your own values.
action_to_steer = [0, 21, -21]

# Vector of the mean score.
scores = []

# Initialising the app for the first time.
is_initialised = False

# Used to toggle between the rock obstacles and the fish colonies.
is_rock = True

# Initialising the introduction.
def welcome_message():
    
    print("\n"*100)

    print("\n\n\nWelcome to MecClearen AI Robotics!")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nHello marine biologist!")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nThere is a big pile of plastic debris in the northeastern part of this ocean threatening marine life.")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nWe present you with the all-new MecClearen robot powered by Artificial Intelligence to navigate the wild ocean without disturbing the colonies of fish and will avoid many random rock obstacles.") 
    raw_input("[Press any key to continue]")
    
    print("\n\n\nBut first, you must train this robot to make round trips from the rubbish collection point to the plastic debris.")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nTo do this, you must challenge the robot with random obstacles that you can draw on the training simulation.")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nWhile training, you can save the trained brain to load it for later use.")
    raw_input("[Press any key to continue]")
    
    print("\n\n\nGood Luck!")
    
    print("\n"*100)
    
# Initialising the global variables and displaying the introduction.
def init():
    
    # Obstacle is an array of pixels on the graphical interface. 
    # If there is a 1 on the array, that means there is an obstacle at that pixel. 
    # If there are no obstacles, then the pixel will be represented with a 0.
    global obstacle
    
    #These are the x and y co-ordinates of the robot's goal.
    # This is either the co-ordinates of the plastic debris or the co-ordinates of the collection point. 
    global goal_x
    global goal_y
    
    global is_initialised
    
    # Initialise the goals of the robot.
    # The goal to reach is at the upper right of the map and lower left corner of the map.
    # I gave 20 pixel as a safety margin.
    goal_x = map_width - 20
    goal_y = map_height - 20
    
    # Creating a NumPy array of the obstacles. 
    # At the start, there aren't any obstacles, therefore making the entire array composed of zeroes.
    obstacle = np.zeros((map_width, map_height))
    
    # This displays the short introduction at the start.
    welcome_message()
    
    # This ensures that the GUI only initialises once.
    is_initialised = True

# Creating the robot class.
class Robot(Widget):
    
    # Initialise the robot's angle (this is the angle between the x-axis and the robot's axis).
    angle = NumericProperty(0) 
    
    # Initialise the robot's rotations 
    # This allows the robot to:
    #    rotate 0 degrees (move forward),
    #    rotate -21 degrees (turn left), or
    #    rotate +21 degrees (turn right).
    rotation = NumericProperty(0)
    
    # Initialise x co-ordinate of the velocity vector for the robot.    
    velocity_x = NumericProperty(0)
    
    # Initialise y co-ordinate of the velocity vector for the robot.
    velocity_y = NumericProperty(0)
    
    # Initialise the robot's velocity vector.
    # This uses velocity_x and velocity_y as the x and y co-ordinates.
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    
    # Initialising the x and y co-ordinates of the centre sensor.
    sensor_centre_x = NumericProperty(0)
    sensor_centre_y = NumericProperty(0)
    # The vector of the centre sensor
    sensor_centre = ReferenceListProperty(sensor_centre_x, sensor_centre_y)
    
    # Initialising the x and y co-ordinates of the left sensor.
    sensor_left_x = NumericProperty(0)
    sensor_left_y = NumericProperty(0)
    # The vector of the left sensor
    sensor_left = ReferenceListProperty(sensor_left_x, sensor_left_y)
    
    # Initialising the x and y co-ordinates of the right sensor.
    sensor_right_x = NumericProperty(0)
    sensor_right_y = NumericProperty(0)
    # The vector of the right sensor
    sensor_right = ReferenceListProperty(sensor_right_x, sensor_right_y)
    
    # Initialising the input signals from the sensors. 
    signal_centre = NumericProperty(0)
    signal_left = NumericProperty(0)
    signal_right = NumericProperty(0)

    def move(self, rotation):
        
        # Update the position of the robot.
        # This uses the robot's previous velocity and position.
        self.pos = Vector(*self.velocity) + self.pos
        
        # This is the rotatio =n of the robot.
        self.rotation = rotation
        
        # This updates the angle of turn (of the robot).
        self.angle = self.angle + self.rotation
        
        # Updating the positions of the sensors.
        self.sensor_centre = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor_right  = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.sensor_left   = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        
        # Calculating the density of the obstacles around the sensors.
        self.signal_centre = int(np.sum(obstacle[int(self.sensor_centre_x)-10:int(self.sensor_centre_x)+10, int(self.sensor_centre_y)-10:int(self.sensor_centre_y)+10]))/400.
        self.signal_left   = int(np.sum(obstacle[int(self.sensor_left_x)-10:int(self.sensor_left_x)+10, int(self.sensor_left_y)-10:int(self.sensor_left_y)+10]))/400.
        self.signal_right  = int(np.sum(obstacle[int(self.sensor_right_x)-10:int(self.sensor_right_x)+10, int(self.sensor_right_y)-10:int(self.sensor_right_y)+10]))/400.
        
        # Check if the sensor is out of the map.
        if self.sensor_centre_x > map_width - 10 or self.sensor_centre_x < 10 or self.sensor_centre_y > map_height - 10 or self.sensor_centre_y < 10:
            # This makes the density of the obstacles very high, making sure that the robot doesn't wander off the map.
            self.signal_centre = 1.
        
        if self.sensor_left_x > map_width - 10 or self.sensor_left_x < 10 or self.sensor_left_y > map_height - 10 or self.sensor_left_y < 10:
            self.signal_left = 1.
        
        if self.sensor_right_x>map_width - 10 or self.sensor_right_x < 10 or self.sensor_right_y > map_height-10 or self.sensor_right_y<10:
            self.signal_right = 1.

class Sensor_Centre(Widget):
    # This is the centre sensor.
    # Note: This class is defined in the robot.kv file.
    pass

class Sensor_Left(Widget):
    # This is the left sensor.
    # Note: This class is defined in the robot.kv file.
    pass

class Sensor_Right(Widget):
    # This is the right sensor.
    # Note: This class is defined in the robot.kv file.
    pass

# The MecClearen class.
class MecClearen(Widget):

    # The robot object from the kivy file.
    robot = ObjectProperty(None)
    
    # The sensor objects from the kivy file.
    sensor_centre = ObjectProperty(None)
    sensor_left = ObjectProperty(None)
    sensor_right = ObjectProperty(None)

    # Start the robot when the simulation is launched.
    def start_robot(self):
        # Start the robot in the center.
        self.robot.center = self.center
        
        # Global current velocity.
        global current_velocity
        
        # Set the initial velocity to current_velocity.
        self.robot.velocity = Vector(current_velocity, 0)

    # Update the robot at each discrete time 't' when reaching a new state
    def update_robot(self, time):

        # The global x and y co-ordinates of the goal.
        global goal_x
        global goal_y
        
        # The global width and height of the map (ocean)
        global map_width
        global map_height
        
        # The global margin offset
        global margin_offset

        # The global AI brain.
        global ai_brain
        # The global last reward.
        global last_reward
        
        # Global current velocity.
        global current_velocity
        
        # Global scores which is the means of the rewards.
        global scores
        # Global last distance. The distance between the robot and the goal (or destination).
        global last_distance

        # Initialise the width and height of the map.
        map_width = self.width
        map_height = self.height

        if not is_initialised:
            # This initialises the map (only once).
            init()

        # Difference of x-coordinates between the goal and the robot
        x_difference = goal_x - self.robot.x
        # Difference of y-coordinates between the goal and the robot
        y_difference = goal_y - self.robot.y
        # Orientiation of the robot with respect to the goal (if the robot is heading perfectly towards the goal, then orientation = 0)
        orientation = Vector(*self.robot.velocity).angle((x_difference, y_difference)) / 180.
        
        # Input state vector, composed of the three signals received by the three sensors, plus the +orientation and -orientation.
        last_signal = [self.robot.signal_centre, self.robot.signal_left, self.robot.signal_right, orientation, -orientation]

        # The action from the AI brain (Deep Q Network class).
        action = ai_brain.update(last_reward, last_signal)
        
        # Adding the score (mean of the last 100 rewards).
        scores.append(ai_brain.score())
        
        # Mapping the action (0, 1 or 2) to the rotation or steering (0 degrees, 21 degrees or -21 degrees).
        rotation = action_to_steer[action]
        
        # Steer the robot depending on the rotation angle.
        self.robot.move(rotation)
        
        # Calculating the distance using the formula: 
        # [square root((x2-x1)squared + (y2 - y1)squared)].
        distance = np.sqrt((self.robot.x - goal_x)**2 + (self.robot.y - goal_y)**2)
        
        # Adjusting the position of sensors after steering the robot to a new position.
        self.sensor_centre.pos = self.robot.sensor_centre
        self.sensor_left.pos   = self.robot.sensor_left
        self.sensor_right.pos  = self.robot.sensor_right

        # Check if the robot is on an obstacle.
        if obstacle[int(self.robot.x),int(self.robot.y)] > 0:
            # Slow down the robot (with speed of 1).
            self.robot.velocity = Vector(1, 0).rotate(self.robot.angle)
            # Give a penalty of -1 as reward.
            last_reward = -1
        else: # Otherwise
            # Set the speed back to current_velocity (to let the robot move at the normal speed).
            self.robot.velocity = Vector(current_velocity, 0).rotate(self.robot.angle)
            # Give it a living penalty of -0.2.
            last_reward = -0.2
            # Check the distance and if the robot gets closer to the goal then give it a positive reward of +0.1.
            if distance < last_distance:
                last_reward = 0.1
        
        # Check if the robot's x is less than 15 pixels from the left edge
        if self.robot.x < margin_offset:
            # Set the x co-ordinate back to 15 pixels from the left edge.
            self.robot.x = margin_offset
            # Give the robot a negative reward of -1. 
            last_reward = -1
            
        # Check if the robot's x is greater than 15 pixels from the right edge
        if self.robot.x > self.width - margin_offset:
            # Set the x co-ordinate back to 15 pixels from the right edge.
            self.robot.x = self.width - margin_offset
            # Give the robot a negative reward of -1. 
            last_reward = -1
            
        # Check if the robot's y is less than 15 pixels from the lower edge
        if self.robot.y < margin_offset:
            # Set the x co-ordinate back to 15 pixels from the lower edge.
            self.robot.y = margin_offset
            # Give the robot a negative reward of -1. 
            last_reward = -1
            
        # Check if the robot's y is greater than 15 pixels from the upper edge
        if self.robot.y > self.height - margin_offset:
            # Set the x co-ordinate back to 10 pixels from the upper edge.
            self.robot.y = self.height - margin_offset
            # Give the robot a negative reward of -1. 
            last_reward = -1

        # If the robot is close to its destination (less than 100 pixels away)
        if distance < 100:
            # Switch the destination.
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
            
        # Set the last distance of the robot from the goal
        last_distance = distance

# Adding the painting tools.
class DrawObstacle(Widget):

    # Draw obstacle when the mouse is pressed down.
    def on_touch_down(self, touch):
        
        global is_rock
        
        if is_rock:
            global length, n_points, last_x, last_y
            with self.canvas:
                # Set the rock's color.
                Color(0.8, 0.6, 0.4)
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                obstacle[int(touch.x), int(touch.y)] = 1
                
        elif is_rock == False:
            global length, n_points, last_x, last_y
            with self.canvas:
                # Set the fish's color.
                Color(1, 0, 0)
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                obstacle[int(touch.x), int(touch.y)] = 1
            
    # Draw more obstacle when the mouse is pressed down and moved simultaneously.
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':

            try:
                touch.ud['line'].points += [touch.x, touch.y]
                x = int(touch.x)
                y = int(touch.y)
                length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
                n_points += 1.
                density = n_points/(length)
                touch.ud['line'].width = int(20 * density + 1)
                obstacle[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
                last_x = x
                last_y = y
            except:
                pass
            finally:
                pass

# Creating the UI elements,
# such as the Buttons (clear, save, load, obstacle toggle and score).
class RobotApp(App):

    def build(self):
        # Assign the parent to the instance of the MecClearen class.
        parent = MecClearen()
        
        # Start the MecClearen robot.
        parent.start_robot()
        
        # Update the status every second.
        Clock.schedule_interval(parent.update_robot, 1.0 / 60.0)
        
        # Initialise the painter with the instance of the DrawObstacle class.
        self.painter = DrawObstacle()

        # Create and position the buttons.
        button_fish  = Button(text = 'Fish',       pos = (670, 30),  size = (200, 100))
        button_rock  = Button(text = 'Rock',       pos = (880, 30),  size = (200, 100))
        button_clear = Button(text = 'Clear Map',  pos = (1120, 30), size = (200, 100))
        button_save  = Button(text = 'Save Brain', pos = (1330, 30), size = (200, 100))
        button_load  = Button(text = 'Load Brain', pos = (1540, 30), size = (200, 100))
        button_score = Button(text = 'Score',      pos = (1780, 30), size = (200, 100))

        # Create the slider to adjust the speed of the robot.
        slider_velocity = Slider(min=3, max=12, value=6, step=1, 
                                 pos = (1700, 150), size = (300, 100), value_track=True, 
                                 value_track_color=[0.1607, 0.4313, 0.0039, 1])
        slider_velocity.bind(value = self.on_value_change_velocity)

        # Create the slider to adjust the softmax temperature.
        slider_temperature = Slider(min=0, max=1000, value=1000, step=10, 
                                 pos = (1400, 150), size = (300, 100), value_track=True, 
                                 value_track_color=[0.0000, 0.6627, 0.8000, 1])
        slider_temperature.bind(value = self.on_value_change_softmax_temperature)

        # Bind the event handler.
        button_fish.bind(on_release  = self.draw_fish)
        button_rock.bind(on_release  = self.draw_rock)
        button_clear.bind(on_release = self.clear_canvas)
        button_save.bind(on_release  = self.save_brain)
        button_load.bind(on_release  = self.load_brain)
        button_score.bind(on_release = self.display_score)

        # Create lables for the end points (destination / goal) and sliders.
        label_start    = Label(text = "Collection Base", pos = (10, 30),     size = (300, 50)) 
        label_end      = Label(text = "Ocean Plastic",   pos = (1700, 1500), size = (300, 50)) 
        label_brain    = Label(text = "Brain Scale",     pos = (1400, 240),  size = (300, 50)) 
        label_velocity = Label(text = "Velocity",        pos = (1700, 240),  size = (300, 50)) 
        
        # Add all the widgets to the parent.
        parent.add_widget(self.painter)

        parent.add_widget(slider_velocity)
        parent.add_widget(slider_temperature)

        parent.add_widget(button_fish)
        parent.add_widget(button_rock)

        parent.add_widget(button_save)
        parent.add_widget(button_load)
        parent.add_widget(button_clear)
        parent.add_widget(button_score)

        parent.add_widget(label_start)
        parent.add_widget(label_end)

        parent.add_widget(label_velocity)
        parent.add_widget(label_brain)

        # Return the parent object.
        return parent
        
    def on_value_change_softmax_temperature(self, instance, value):
        # Assign the slider value to the softmax_temperature
        ai_brain.softmax_temperature = value
        
    def on_value_change_velocity(self, instance, value):
        global current_velocity
        # Set the value of the slider to the current_velocity.
        # Increasing the velocity may rempt the robot to cross thin obstacles.
        current_velocity = value

    # Event handler for drawing rocks.
    def draw_rock(self, obj):
        global is_rock
        is_rock = True

    # Event handler for drawing fish hotspot / schools of fish.
    def draw_fish(self, obj):
        # Toggle the global is_rock flag (set it to False).
        global is_rock
        is_rock = False
        
    # Event handler for clearing the canvas (map).
    def clear_canvas(self, obj):
        global obstacle
        # Clear the canvas.
        self.painter.canvas.clear()
        # Clear all the obstacles and reset the array with Numpy zeros. 
        obstacle = np.zeros((map_width, map_height))
   
    # Event handler for saving the brain.
    def save_brain(self, obj):
        print("Saving robot's brain...")
        # Save the AI brain.
        ai_brain.save()
        # Initialise the plot with scores array.
        plot.plot(scores)
        # Display the scores in a graph in the console.
        plot.show()
    
    # Event handler for loading the last saved brain back in.
    def load_brain(self, obj):
        print("Loading the last saved robotic brain...")
        # Load the AI brain.
        ai_brain.load()
 
    # Event handler for displaying the current score.
    def display_score(self, obj):
        print("\nCurrent Score: {0}".format(ai_brain.score()))
        
# Run the simulation.
if __name__ == '__main__':
    Window.size = (1000, 850)
    print("\n" * 100)
    RobotApp().run()