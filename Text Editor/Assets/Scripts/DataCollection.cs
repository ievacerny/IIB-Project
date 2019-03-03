using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System;
using UnityEngine;
using Leap;
using Leap.Unity;

public class DataCollection : MonoBehaviour {

    LeapProvider provider;
    int frame_counter = 0;
    StreamWriter writer;
    string path;



    // Use this for initialization
    void Start () {
        path = "../Database/MyDatabase/data_" + DateTime.Now.Ticks + ".csv";
        provider = FindObjectOfType<LeapProvider>() as LeapProvider;
        writer = new StreamWriter(path);
    }
	
	// Update is called once per frame
	void Update () {

        int idx = frame_counter % 1;

        if (idx == 0)
        {
            Frame frame = provider.CurrentFrame;
            if (frame.Hands.Count != 0)
            {
                Hand hand = frame.Hands[0];
                Vector direction = hand.Direction;
                Vector position = hand.PalmPosition;
                Vector palm_normal = hand.PalmNormal;
                Vector palm_velocity = hand.PalmVelocity;
                Vector wrist_position = hand.WristPosition;
                LeapQuaternion rotation = hand.Rotation;

                Vector[] f_directions = new Vector[5];
                Vector[] f_positions = new Vector[5];
                Vector[] b_joint_beg = new Vector[20];
                Vector[] b_directions = new Vector[20];
                LeapQuaternion[] b_rotations = new LeapQuaternion[20];
                int f_idx = 0;
                int b_idx = 0;
                foreach (Finger finger in hand.Fingers)
                {
                    f_directions[f_idx] = finger.Direction;
                    f_positions[f_idx] = finger.TipPosition;
                    foreach (Bone bone in finger.bones)
                    {
                        b_joint_beg[b_idx] = bone.PrevJoint;
                        b_directions[b_idx] = bone.Direction;
                        b_rotations[b_idx] = bone.Rotation;
                        b_idx++;
                    }
                    f_idx++;
                }
                
                string printable = "";
                printable += WriteVector(position); // 0:3 - hand position
                printable += "," + WriteVector(wrist_position); // 3:6 - wrist position
                foreach (var f_pos in f_positions) //6:21 - fingertip positions from thumb to pinky
                {
                    printable += "," + WriteVector(f_pos);
                }
                printable += "," + WriteVector(direction); // 21:24 - hand direction (from palm to fingers)
                printable += "," + WriteVector(palm_normal); // 24:27 - palm normal
                printable += "," + WriteVector(palm_velocity); // 27:30 - palm_velocity - zero?
                printable += "," + WriteQuaternion(rotation); // 30:34 - hand rotation
                foreach (var f_dir in f_directions) // 34:49 - fingertip positions from thumb to pinky
                {
                    printable += "," + WriteVector(f_dir);
                }
                for(int i=0; i<20; i++)
                {
                    printable += "," + WriteVector(b_joint_beg[i]); // 49:52 bone beginning
                    printable += "," + WriteVector(b_directions[i]); // 52:55 bone direction
                    printable += "," + WriteQuaternion(b_rotations[i]); // 55:59 bone rotation

                }

                writer.WriteLine(printable);
            }
        }
        frame_counter++;
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application ending after " + Time.time + " seconds. File " + path);
        writer.Close();
    }

    string WriteVector(Vector v)
    {
        return(v.x + "," + v.y + "," + v.z);
    }

    string WriteQuaternion(LeapQuaternion q)
    {
        return (q.w + "," + q.x + "," + q.y + "," + q.z);
    }
}

