﻿using Leap;
using Leap.Unity;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class HandControl : MonoBehaviour
{
    public enum GRstate { NOTHING, ML, SELECTION_ACTIVATE, SELECTION_DEACTIVATE };

    #region Parameters
    [SerializeField] private bool record_data = false;
    private readonly int frame_step = 8;
    private string file_path = "../Database/MyDatabase/data_";
    #endregion
    #region Private Attributes
    [SerializeField] private GRstate current_state = GRstate.NOTHING;
    private TCPClient client;
    private bool connection_active = false;
    private LeapProvider provider;
    private Hand hand = null;
    private FocusControl focus;
    private GameObject current_page;
    private PageView current_page_script = null;
    private StreamWriter writer = null;
    #endregion

    #region Unity Functions

    void Start()
    {
        client = new TCPClient();
        connection_active = client.IsClientConnected();
        provider = GetComponent<LeapProvider>();
        focus = GetComponent<FocusControl>();
        current_page = focus.GetActivePage();
        if (current_page != null)
            current_page_script = current_page.GetComponent<PageView>();

        if (record_data)
        {
            file_path = file_path + DateTime.Now.Ticks + ".csv";
            writer = new StreamWriter(file_path);
        }
    }
	
	void Update()
    {
        // Update page and page script references
        GameObject new_page = focus.GetActivePage();
        if (!GameObject.ReferenceEquals(new_page, current_page))
        {
            current_page = new_page;
            if (current_page != null)
                current_page_script = current_page.GetComponent<PageView>();
            else
                current_page_script = null;
        }

        // If no page is active, no need to look for the hand
        if (current_page == null)
        {
            current_state = GRstate.NOTHING;
        }
        else
        {
            // Determine the state of the recognition network (and possibly record data)
            hand = FindRightHand(provider.CurrentFrame);
            if (hand == null)
            {
                if (current_page_script.selection_mode)
                    current_page_script.RemoveSelection();
                current_page_script.selection_mode = false;
                current_page_script.trigger_active = false;
                current_state = GRstate.NOTHING;
            }
            else
            {
                if (current_page_script.trigger_active && !current_page_script.selection_mode)
                    current_state = GRstate.SELECTION_ACTIVATE;
                else if (current_page_script.trigger_active && current_page_script.selection_mode)
                    current_state = GRstate.SELECTION_DEACTIVATE;
                else if (!current_page_script.trigger_active && !current_page_script.selection_mode)
                    current_state = GRstate.ML;
                else
                    Debug.LogWarning("Unexpected gesture recognition activation state");

                if (record_data)
                {
                    RecordFrame();
                    return;
                }
            }
        }

        // Take care of selection and deselection
        switch (current_state)
        {
            case GRstate.SELECTION_ACTIVATE:
                CheckSelectionActivationGesture();
                break;

            case GRstate.SELECTION_DEACTIVATE:
                CheckSelectionDeactivationGesture();
                break;

            default:
                break;
        }

        // Every n-th frame actually communicate with the server and update the gesture code
        if (Time.frameCount % frame_step == 0)
        {
            SendGestureData();
            int gesture_code = ReadGestureCode();
            if (current_state != GRstate.ML)
                current_page_script.gesture_code = 0;
            else
                current_page_script.gesture_code = gesture_code;
        }
        // In between server communications, keep the same ML gesture code or set to 0 if the state has changed
        else
        {
            if (current_state != GRstate.ML && current_page_script.gesture_code != 0)
                current_page_script.gesture_code = 0;
        }
    }

    void OnApplicationQuit()
    {
        if (record_data)
        {
            Debug.Log("Application ending after " + Time.time + " seconds. File " + file_path);
            writer.Close();
        }
    }

    #endregion

    #region Private Methods

    private Hand FindRightHand(Frame frame)
    {
        if (frame.Hands.Count == 0)
            return null;
        Hand right_hand = null;
        foreach (Hand any_hand in frame.Hands)
        {
            if (any_hand.IsRight)
            {
                right_hand = any_hand;
                break;
            }
        }
        return right_hand;
    }

    private void SendGestureData()
    {
        if (!connection_active)
            return;

        try
        {
            if (current_state == GRstate.ML)
                client.SendMessage(EncodeFrameData());
            else
                client.SendMessage("0");
        }
        catch (Exception e)
        {
            Debug.LogError(string.Format("Exception: {0}", e));
            connection_active = false;
        }
    }

    private int ReadGestureCode()
    {
        if (!connection_active)
            return 0;

        string message = "";

        try
        {
            message = client.ReadMessage();
        }
        catch (Exception e)
        {
            Debug.LogError(string.Format("Exception: {0}", e));
            connection_active = false;
            return 0;
        }

        if (message.Length == 1)
        {
            return Convert.ToInt32(message);
        }
        else
        {
            Debug.LogWarning("Received unexpected message: " + message);
            return 0;
        }
    }

    private void CheckSelectionActivationGesture()
    {
        List<Finger> fingers = hand.Fingers;
        bool condition = (
            fingers[0].IsExtended && fingers[1].IsExtended && fingers[2].IsExtended &&
            !fingers[3].IsExtended && !fingers[4].IsExtended
        );
        if (condition)
        {
            current_page_script.ChangeSelectionModeStatus(true);
            current_state = GRstate.SELECTION_DEACTIVATE;
        }
    }

    private void CheckSelectionDeactivationGesture()
    {
        List<Finger> fingers = hand.Fingers;
        bool thumb_condition = (
            !fingers[0].IsExtended && fingers[1].IsExtended && fingers[2].IsExtended &&
            !fingers[3].IsExtended && !fingers[4].IsExtended
        );
        bool activation_condition = (
            fingers[0].IsExtended && fingers[1].IsExtended && fingers[2].IsExtended &&
            !fingers[3].IsExtended && !fingers[4].IsExtended
        );
        if (thumb_condition)
        {
            current_page_script.ChangeSelectionModeStatus(false);
            current_page_script.trigger_active = false;
            current_state = GRstate.ML;
        }
        else if (!activation_condition)
        {
            current_page_script.ChangeSelectionModeStatus(false, error: true);
            current_page_script.trigger_active = false;
            current_page_script.RemoveSelection();
            current_state = GRstate.ML;
        }
    }

    private string EncodeFrameData()
    {
        string[] data_strings = new string[76];
        Vector position = hand.PalmPosition;
        data_strings[0] = WriteVector(position);
        data_strings[1] = WriteVector(hand.WristPosition-position);
        int bone_str_idx = 16;
        for (int i=2; i<7; i++)
        {
            data_strings[i] = WriteVector(hand.Fingers[i - 2].TipPosition - position);
            data_strings[i + 9] = WriteVector(hand.Fingers[i - 2].Direction);
            foreach (Bone bone in hand.Fingers[i - 2].bones)
            {
                data_strings[bone_str_idx] = WriteVector(bone.PrevJoint - position);
                data_strings[bone_str_idx + 1] = WriteVector(bone.Direction);
                data_strings[bone_str_idx + 2] = WriteQuaternion(bone.Rotation);
                bone_str_idx += 3;
            }
        }
        data_strings[7] = WriteVector(hand.Direction);
        data_strings[8] = WriteVector(hand.PalmNormal);
        data_strings[9] = WriteVector(hand.PalmVelocity);
        data_strings[10] = WriteQuaternion(hand.Rotation);

        return string.Join(",", data_strings);
    }

    private string WriteVector(Vector v)
    {
        return (v.x + "," + v.y + "," + v.z);
    }

    private string WriteQuaternion(LeapQuaternion q)
    {
        return (q.w + "," + q.x + "," + q.y + "," + q.z);
    }

    private void RecordFrame()
    {
        writer.WriteLine(EncodeFrameData());
    }
    #endregion
}
