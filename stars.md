---
classes: wide
header:
  overlay_image: /images/stars-images/stars.jpg

title: Stars Simulation
toc: true
toc_label: "Overview"

---

<style type="text/css">
body {
  font-size: 13pt;
}

pre {
  background-color: white;
}

code {
  background-color: white;
}
</style>

## Stars Simulation

This program simulates the motion of astronomical bodies in space under the influence of gravitational forces. It uses the laws of physics to calculate the positions and velocities of the bodies over time. The simulation is visualized using a graphical user interface (GUI) that displays the bodies' movements on a canvas.

## NBody.java

## Class Overview
We will create a file by calling it `NBody.java` which defines the `NBody` class that represents an astronomical body. It contains methods for drawing the stars on a canvas and updating their positions based on physical forces.

### Libraries needed
Let's first import the following libraries that will be required for this to work.
~~~ java
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Random;
~~~

With these libraries, we will be able to draw our stars and show their movement.

### Instance Variables
Now we are gonna need some instance variables for our stars.

~~~ java
public class NBody extends Canvas implements ActionListener {
	public int n;
	public int x;
	public int y;
	public int diameter;
	public double dt;
	public double mass;
	public double xVelocity;
	public double yVelocity;
	public double xForce;
	public double yForce;
	public int xCoordinate;
	public int yCoordinate;
	public Color color;
	public double maxVel;
	public double maxMass;
	public int panelSize;

    public List<NBody> listOfStars;
	public double scalingFactor;
	public double G = 6.673e-11;
~~~

- `n`, `x`, `y`: Integers representing coordinates and size of the body.
- `diameter`: Integer representing the diameter of the body.
- `dt`: Double representing the time step for the simulation.
- `mass`: Double representing the mass of the body.
- `xVelocity`, `yVelocity`: Doubles representing the velocity components of the body.
- `xForce`, `yForce`: Doubles representing the net force components acting on the body.
- `xCoordinate`, `yCoordinate`: Integers representing the current position of the body.
- `color`: Color object representing the color of the body.
- `maxVel`, `maxMass`: Doubles representing the maximum velocity and mass values.
- `panelSize`: Integer representing the size of the canvas panel.
- `listOfStars`: List of `NBody` objects representing all the bodies in the simulation.
- `scalingFactor`: Double representing the scaling factor for the gravitational forces.
- `G`: Constant double representing the gravitational constant.

All these variables will be necessary for calculating the movement of the stars.

### Constructors

Next, we are going to need a constructor to actually create star objects with their information.

~~~ java

public NBody(double mass, int xCoordinateValue, int yCoordinateValue, double xVelocity, double yVelocity,
	int diameter) {
	this.mass = mass;
	this.xCoordinate = xCoordinateValue;
	this.yCoordinate = yCoordinateValue;
	this.xVelocity = xVelocity;
	this.yVelocity = yVelocity;
	this.diameter = diameter;

	Random r = new Random();
	color = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256), r.nextInt(256));
}

public NBody(List<NBody> starsList, double sc) {
	listOfStars = starsList;
	scalingFactor = sc;
}
~~~

- `NBody(double mass, int xCoordinateValue, int yCoordinateValue, double xVelocity, double yVelocity, int diameter)`: Constructs an `NBody` object with the specified properties, such as mass, position, velocity, and diameter.
- `NBody(List<NBody> starsList, double sc)`: Constructs an `NBody` object that takes a list of `NBody` objects and a scaling factor as parameters.

These constructors now contain all the information our star needs to calculate its movement.

### Methods
Now, we are going to need the following methods in order to calculate the star's movements and draw them on the screen.

~~~ java
public void drawCircle(Graphics g, int centerX, int centerY, int r) {
	int diameter = 2 * r;
	g.fillOval(centerX - r, centerY - r, diameter, diameter);
}
~~~

- `drawCircle(Graphics g, int centerX, int centerY, int r)`: Draws a circle on the canvas at the specified center coordinates and radius.

~~~ java
public void paint(Graphics g) {
	//calling super to do initialization
	super.paint(g);

	for (int i = 0; i < listOfStars.size(); i++) {
		
		//setting the color of the graphics based on the random color of the stars
		g.setColor(listOfStars.get(i).getColor());
		
		//drawing the star
		g.fillOval(listOfStars.get(i).getxCoordinate(), listOfStars.get(i).getyCoordinate(), listOfStars.get(i).getNBodySize(),
				listOfStars.get(i).getNBodySize());
	}
}
~~~

- `paint(Graphics g)`: Overrides the `paint` method to draw the bodies on the canvas using the `Graphics` object.

~~~ java
public void actionPerformed(ActionEvent e) {
	//updating the stars and repainting the screen
	update();
	repaint();
	Toolkit.getDefaultToolkit().sync();
}
~~~

- `actionPerformed(ActionEvent e)`: Overrides the `actionPerformed` method from `ActionListener` interface to handle the update and repainting of the bodies on the canvas.

~~~ java
// helper methods to calculate the physics of the stars
public void update() {
	int numberOfStars;
	for (numberOfStars = 0; numberOfStars < listOfStars.size() - 1; numberOfStars++) {
		listOfStars.get(numberOfStars).force(listOfStars.get(numberOfStars + 1), scalingFactor);
		listOfStars.get(numberOfStars).updatePos();
		listOfStars.get(numberOfStars).resetForce();
	}
}
~~~

- `update()`: Updates the positions and forces of the bodies based on the gravitational interactions between them.

~~~ java
public void updatePos() {
	// update x and y velocities
	xVelocity += xForce / mass;
	yVelocity += yForce / mass;

	// update x and y coordinates
	xCoordinate += xVelocity;
	yCoordinate += yVelocity;
}
~~~

- `updatePos()`: Updates the velocities and positions of the body based on the net forces acting on it.

~~~ java
// calculate force
public void force(NBody nBody, double scale) {
	NBody currentNBody = this;

	double xCoordinate = nBody.xCoordinate - currentNBody.xCoordinate;
	double yCoordinate = nBody.yCoordinate - currentNBody.yCoordinate;
	
	double magnitude = Math.sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
	
	// calculate the graviation force between two masses
	double force = (G * currentNBody.mass * nBody.mass / ((magnitude * magnitude) / scale));
	
	currentNBody.xForce += force * xCoordinate / magnitude;
	currentNBody.yForce += force * yCoordinate / magnitude;
}
~~~
- `force(NBody nBody, double scale)`: Calculates the gravitational force between the current body and another body.

~~~ java
public void resetForce() {
	xForce = 0;
	yForce = 0;
}
~~~
- `resetForce()`: Resets the net force components of the body.

### Getters

We will now need the following getters to access our star's information when necessary

~~~ java

public void resetForce() {
	xForce = 0;
	yForce = 0;
}

public double getMass() {
	return mass;
}

public int getxCoordinate() {
	return xCoordinate;
}

public int getyCoordinate() {
	return yCoordinate;
}

public double getxVelocity() {
	return xVelocity;
}

public double getyVelo() {
	return yVelocity;
}

public int getNBodySize() {
	return diameter;
}

public Color getColor() {
	return color;
}
~~~

Now that our `NBody` class is complete, we need to create the graphical user interface (GUI) of the stars moving.

## NBodyApp.java

## Class Overview
The `NBodyApp` class sets up the graphical user interface (GUI) and manages the main logic of the stars simulation using the `NBody` class.

## Libraries needed
Let's import the following libraries required for this to work.
~~~ java
import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.Timer;
~~~

### Instance Variables
We are going to need some instance variables to create the star objects.

~~~ java
public class NBodyApp {
	public static List <NBody> tempList = null;
	public static double scalingFactor = 0;
	public static int n = 20;
~~~

- `tempList`: List of `NBody` objects representing temporary bodies used in the simulation.
- `scalingFactor`: Double representing the scaling factor for the gravitational forces.
- `n`: Integer representing the number of bodies in the simulation.

### Method Required

We need to create a method to actually generate the stars.

~~~ java
public static void GenerateNBodiesBasedOnUsersNumber(int numberOfNBodiesToGenerate) {
	scalingFactor = 1000.0;
	Random r = new Random();
	int lowDiameter = 10;
	int highDiameter = 20;

	int lowMass = 1800;
	int highMass = 300000;

	int lowXcoordinate = 20;
	int highXCoordinate = 780;

	int lowYCoordinate = 20;
	int highYCoordinate = 780;

	for (int i = 0; i < numberOfNBodiesToGenerate; i++) {
		tempList.add(new NBody(r.nextInt(highMass - lowMass) + lowMass,
		r.nextInt(highXCoordinate - lowXcoordinate) + lowXcoordinate,
		r.nextInt(highYCoordinate - lowYCoordinate) + lowYCoordinate, r.nextDouble(-0.001, 0.001), r.nextDouble(-0.001, 0.001),
		r.nextInt(highDiameter - lowDiameter) + lowDiameter));
	}
}

~~~

This method will generate a certain number of stars based on the user input.

**Parameters:**
- `numberOfNBodiesToGenerate`: An integer representing the desired number of `NBody` objects to generate.

**Method Description:**
- Sets the scaling factor for gravitational forces to `1000.0`.
- Initializes variables for the lower and upper limits of diameter, mass, and coordinates.
- Creates a new instance of the `Random` class to generate random values.
- Uses the `nextInt()` method of the `Random` class to generate random values within the specified ranges for each attribute of the `NBody` objects.
- Creates new `NBody` objects with the generated values and adds them to the `tempList` list.

## Putting it all together

Finally, we need to create the `main method` to create the GUI of the stars.

~~~ java
public static void main(String[] args) {
	// user input
	try {
		n = Integer.parseInt(args[0]);
		n=n>20?n=20:Integer.parseInt(args[0]);
	}
	catch (Exception e) {
		// default num of stars to 20 if the user runs it without using the Command Prompt Line.
		System.out.println("You entered invalid value at command line");
	}
	tempList = new ArrayList<NBody>();
	GenerateNBodiesBasedOnUsersNumber(n);

	NBody nbody = new NBody(tempList, scalingFactor);
	nbody.setBackground(Color.black);

	JFrame frame = new JFrame();
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	nbody.setBackground(Color.BLACK);
	nbody.panelSize = 800;
	nbody.maxVel = 10;
	nbody.maxMass = 10;
	nbody.dt = 0.1;
	nbody.setPreferredSize(new Dimension(nbody.panelSize, nbody.panelSize));

	frame.add(nbody);
	frame.pack();

	Timer timer = new Timer(16, nbody);
	timer.start();

	frame.setVisible(true);
}

~~~ 

### The `main method` will do the following:

- Reads user input from the command line. The user can provide the number of stars as a command line argument.
- If a valid integer value is provided, it is assigned to the variable `n`. If the value exceeds 20, `n` is capped at 20.
- If an exception occurs during the parsing of the command line argument, a message is displayed indicating an invalid value was entered.
- Initializes the `tempList` as a new instance of `ArrayList<NBody>`.
- Calls the `GenerateNBodiesBasedOnUsersNumber` method to generate `NBody` objects based on the user input and adds them to the `tempList`.
- Creates a new instance of `NBody` and passes the `tempList` and scaling factor to its constructor.
- Sets the background color of the `nbody` object to black.
- Creates a new `JFrame` object.
- Sets the default close operation of the frame to exit the application when closed.
- Sets the background color of the `nbody` object to black.
- Sets the panel size, maximum velocity, maximum mass, and time step values of the `nbody` object.
- Sets the preferred size of the `nbody` object to create a square panel.
- Adds the `nbody` object to the frame.
- Packs the frame to adjust its size based on the preferred size of the `nbody` object.
- Creates a new `Timer` object that fires an action event every 16 milliseconds and passes the `nbody` object as the ActionListener.
- Starts the timer.
- Sets the visibility of the frame to true, making it visible to the user.

## Demonstration
![](/images/stars-images/stars.gif)

This method is automatically invoked when running the `NBodyApp` program. If the user provides a valid number of stars as a command line argument, it will generate a graphical simulation of the N-Body problem with the specified number of stars. 

## The Full Code

This is all the code together on how it appears on my [Stars-Simulation](https://github.com/samikamal21/Stars-Simulation) repository:

# NBody.java

~~~ java
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import java.util.Random;

public class NBody extends Canvas implements ActionListener {
	public int n;
	public int x;
	public int y;
	public int diameter;
	public double dt;
	public double mass;
	public double xVelocity;
	public double yVelocity;
	public double xForce;
	public double yForce;
	public int xCoordinate;
	public int yCoordinate;
	public Color color;
	public double maxVel;
	public double maxMass;
	public int panelSize;

	public List<NBody> listOfStars;
	public double scalingFactor;
	public double G = 6.673e-11;

	public NBody(double mass, int xCoordinateValue, int yCoordinateValue, double xVelocity, double yVelocity,
		int diameter) {
		this.mass = mass;
		this.xCoordinate = xCoordinateValue;
		this.yCoordinate = yCoordinateValue;
		this.xVelocity = xVelocity;
		this.yVelocity = yVelocity;
		this.diameter = diameter;

		Random r = new Random();
		color = new Color(r.nextInt(256), r.nextInt(256), r.nextInt(256), r.nextInt(256));
	}

	public NBody(List<NBody> starsList, double sc) {
		listOfStars = starsList;
		scalingFactor = sc;
	}

	public void drawCircle(Graphics g, int centerX, int centerY, int r) {
		int diameter = 2 * r;
		g.fillOval(centerX - r, centerY - r, diameter, diameter);
	}

	public void paint(Graphics g) {
		//calling super to do initialization
		super.paint(g);

		for (int i = 0; i < listOfStars.size(); i++) {
			
			//setting the color of the graphics based on the random color of the stars
			g.setColor(listOfStars.get(i).getColor());
			
			//drawing the star
			g.fillOval(listOfStars.get(i).getxCoordinate(), listOfStars.get(i).getyCoordinate(), listOfStars.get(i).getNBodySize(),
					listOfStars.get(i).getNBodySize());
		}
	}

	public void actionPerformed(ActionEvent e) {
		//updating the stars and repainting the screen
		update();
		repaint();
		Toolkit.getDefaultToolkit().sync();
	}

	// helper methods to calculate the physics of the stars
	public void update() {
		int numberOfStars;
		for (numberOfStars = 0; numberOfStars < listOfStars.size() - 1; numberOfStars++) {
			listOfStars.get(numberOfStars).force(listOfStars.get(numberOfStars + 1), scalingFactor);
			listOfStars.get(numberOfStars).updatePos();
			listOfStars.get(numberOfStars).resetForce();
		}
	}

	public void updatePos() {
		// update x and y velocities
		xVelocity += xForce / mass;
		yVelocity += yForce / mass;

		// update x and y coordinates
		xCoordinate += xVelocity;
		yCoordinate += yVelocity;
	}

	// calculate force
	public void force(NBody nBody, double scale) {
		NBody currentNBody = this;
	
		double xCoordinate = nBody.xCoordinate - currentNBody.xCoordinate;
		double yCoordinate = nBody.yCoordinate - currentNBody.yCoordinate;
		
		double magnitude = Math.sqrt(xCoordinate * xCoordinate + yCoordinate * yCoordinate);
		
		// calculate the graviation force between two masses
		double force = (G * currentNBody.mass * nBody.mass / ((magnitude * magnitude) / scale));
		
		currentNBody.xForce += force * xCoordinate / magnitude;
		currentNBody.yForce += force * yCoordinate / magnitude;
	}

	public void resetForce() {
		xForce = 0;
		yForce = 0;
	}

	public double getMass() {
		return mass;
	}

	public int getxCoordinate() {
		return xCoordinate;
	}

	public int getyCoordinate() {
		return yCoordinate;
	}

	public double getxVelocity() {
		return xVelocity;
	}

	public double getyVelo() {
		return yVelocity;
	}

	public int getNBodySize() {
		return diameter;
	}
	
	public Color getColor() {
		return color;
	}
}

~~~

# NBodyApp.java

~~~ java

import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.Timer;

public class NBodyApp {
	public static List <NBody> tempList = null;
	public static double scalingFactor = 0;
	public static int n = 20;

	public static void GenerateNBodiesBasedOnUsersNumber(int numberOfNBodiesToGenerate) {
		scalingFactor = 1000.0;
		Random r = new Random();
		int lowDiameter = 10;
		int highDiameter = 20;

		int lowMass = 1800;
		int highMass = 300000;

		int lowXcoordinate = 20;
		int highXCoordinate = 780;

		int lowYCoordinate = 20;
		int highYCoordinate = 780;

		for (int i = 0; i < numberOfNBodiesToGenerate; i++) {
			tempList.add(new NBody(r.nextInt(highMass - lowMass) + lowMass,
			r.nextInt(highXCoordinate - lowXcoordinate) + lowXcoordinate,
			r.nextInt(highYCoordinate - lowYCoordinate) + lowYCoordinate, r.nextDouble(-0.001, 0.001), r.nextDouble(-0.001, 0.001),
			r.nextInt(highDiameter - lowDiameter) + lowDiameter));
		}
	}

	public static void main(String[] args) {
		// user input
		try {
			n = Integer.parseInt(args[0]);
			n=n>20?n=500:Integer.parseInt(args[0]);
		}
		catch (Exception e) {
			// default num of stars to 20 if the user runs it without using the Command Prompt Line.
			System.out.println("You entered invalid value at command line");
		}
		tempList = new ArrayList<NBody>();
		GenerateNBodiesBasedOnUsersNumber(n);

		NBody nbody = new NBody(tempList, scalingFactor);
		nbody.setBackground(Color.black);

		JFrame frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		nbody.setBackground(Color.BLACK);
		nbody.panelSize = 800;
		nbody.maxVel = 10;
		nbody.maxMass = 10;
		nbody.dt = 0.1;
		nbody.setPreferredSize(new Dimension(nbody.panelSize, nbody.panelSize));

		frame.add(nbody);
		frame.pack();

		Timer timer = new Timer(16, nbody);
		timer.start();

		frame.setVisible(true);
	}
}
~~~
