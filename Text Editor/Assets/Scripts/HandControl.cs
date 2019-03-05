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

    // Use this for initialization
    void Start () {
        provider = FindObjectOfType<LeapProvider>() as LeapProvider;
        focus = GetComponent<FocusControl>();
        current_page = focus.GetActivePage();
        if (current_page != null)
        {
            current_page_script = current_page.GetComponent<PageView>();
        }
    }
	
	// Update is called once per frame
	void Update () {

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

        switch (current_state)
        {
            case GRstate.SELECTION_ACTIVATE:
                CheckSelectionActivationGesture();
                break;

            case GRstate.SELECTION_DEACTIVATE:
                CheckSelectionDeactivationGesture();
                break;

            case GRstate.ML:
                break;

            default:
                break;
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
}
