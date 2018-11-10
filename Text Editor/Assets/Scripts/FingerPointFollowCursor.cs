using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FingerPointFollowCursor : MonoBehaviour {

    private Transform cursor_object_transf;
    private float z_pos;
    [SerializeField] private float step = 0.1f;
    [SerializeField] private Transform left_index;
    //private Leap.Unity.HandModel leapHandModel;
    private bool position_tracking_on = false;

    void Start ()
    {
        cursor_object_transf = GetComponent<Transform>();
        z_pos = cursor_object_transf.position.z - Camera.main.transform.position.z;
        Cursor.visible = false;
	}
	
	void Update ()
    {
        if (position_tracking_on)
        {
            cursor_object_transf.position = left_index.position;
        }
        else
        {
            if (Input.GetKeyDown(KeyCode.PageDown))
                z_pos -= step;
            if (Input.GetKeyDown(KeyCode.PageUp))
                z_pos += step;

            cursor_object_transf.position = Camera.main.ScreenToWorldPoint(
                new Vector3(Input.mousePosition.x, Input.mousePosition.y, z_pos));
        }
    }

    void OnApplicationQuit()
    {
        Cursor.visible = true;
    }

    public void ActivatePositionTracking(GameObject hand)
    {
        Debug.Log("Activate");
        position_tracking_on = true;
    }

    public void DeactivatePositionTracking(GameObject hand)
    {
        Debug.Log("Dectivate");
        position_tracking_on = false;
    }

}
