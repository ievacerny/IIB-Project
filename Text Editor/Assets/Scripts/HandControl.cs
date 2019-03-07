using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Leap;
using Leap.Unity;

public class HandControl : MonoBehaviour {

    public enum GRstate { NOTHING, ML, SELECTION_ACTIVATE, SELECTION_DEACTIVATE };

    LeapProvider provider;
    FocusControl focus;
    GameObject current_page;
    PageView current_page_script = null;
    public GRstate current_state = GRstate.NOTHING;
    Hand hand = null;
    TCPClient client;
    int update_counter = 0;

    // Use this for initialization
    void Start () {
        provider = FindObjectOfType<LeapProvider>() as LeapProvider;
        focus = GetComponent<FocusControl>();
        current_page = focus.GetActivePage();
        if (current_page != null)
        {
            current_page_script = current_page.GetComponent<PageView>();
        }
        client = new TCPClient();
    }
	
	// Update is called once per frame
	void Update () {

        update_counter++;
        if (update_counter%5 != 0)
        {
            return;
        }

        //if (Input.GetKeyDown(KeyCode.Space))
        //{
        //    //Debug.Log("Before entering the send function");
        //    client.SendMessage(EncodeFrameData());
        //    //client.SendMessage("0");
        //    //Debug.Log("After sending the message. Before the reading function");
        //    client.ReadMessage();
        //    //Debug.Log("After reading the message");
        //}

        // Update page references
        GameObject new_page = focus.GetActivePage();
        if (GameObject.ReferenceEquals(new_page, current_page))
        {
            if (current_page != null && current_page_script == null)
            {
                current_page_script = current_page.GetComponent<PageView>();
            }
        }
        else
        {
            current_page = new_page;
            if (current_page != null)
            {
                current_page_script = current_page.GetComponent<PageView>();
            }
            else
            {
                current_page_script = null;
            }
        }

        // If no page is active, no gesture recognition needed
        if (current_page == null)
        {
            current_state = GRstate.NOTHING;
            return;
        }

        // Determine the state of the recognition network
        Frame frame = provider.CurrentFrame;
        if (frame.Hands.Count != 0)
        {
            hand = null;
            // Find a right hand in the frame. If no right hands - return
            foreach(Hand any_hand in frame.Hands)
            {
                if (any_hand.IsRight)
                {
                    hand = any_hand;
                    break;
                }
            }
            if (hand == null)
            {
                if (current_page_script.selection_mode) current_page_script.RemoveSelection();
                current_page_script.selection_mode = false;
                current_page_script.trigger_active = false;
                current_state = GRstate.NOTHING;
                return;
            }

            if (current_page_script.trigger_active && !current_page_script.selection_mode)
            {
                current_state = GRstate.SELECTION_ACTIVATE;
            }
            else if (current_page_script.trigger_active && current_page_script.selection_mode)
            {
                current_state = GRstate.SELECTION_DEACTIVATE;
            }
            else if (!current_page_script.trigger_active && !current_page_script.selection_mode)
            {
                current_state = GRstate.ML;
            }
            else
            {
                Debug.LogWarning("Unexpected gesture recognition activation state");
            }
        }
        else
        {
            hand = null;
            if (current_page_script.selection_mode) current_page_script.RemoveSelection();
            current_page_script.selection_mode = false;
            current_page_script.trigger_active = false;
            current_state = GRstate.NOTHING;
        }

        UpdateWindow();

        switch (current_state)
        {
            case GRstate.SELECTION_ACTIVATE:
                CheckSelectionActivationGesture();
                break;

            case GRstate.SELECTION_DEACTIVATE:
                CheckSelectionDeactivationGesture();
                break;

            case GRstate.ML:
                client.SendMessage(EncodeFrameData());
                client.ReadMessage();
                break;

            default:
                break;
        }
    }


    private void UpdateWindow()
    {
        // TODO: Read data and send it to the model
        return;
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

    private void PredictGesture()
    {
        // Get the prediction of the gesture from the model. Might not be necessary
        //Debug.Log(EncodeFrameData());
        return;
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
}
