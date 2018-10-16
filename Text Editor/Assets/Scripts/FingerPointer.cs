using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FingerPointer : MonoBehaviour {

    public float movement_size = 1.0f;

	void Start ()
    {
		
	}
	
	void Update ()
    {
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            Vector3 position = this.transform.position;
            position.x -= movement_size;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            Vector3 position = this.transform.position;
            position.x += movement_size;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.UpArrow))
        {
            Vector3 position = this.transform.position;
            position.y += movement_size;
            this.transform.position = position;
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            Vector3 position = this.transform.position;
            position.y -= movement_size;
            this.transform.position = position;
        }
        if (Input.GetKeyDown(KeyCode.LeftControl))
        {
            Debug.Log("Down");
            MouseOperations.MouseEvent(MouseOperations.MouseEventFlags.LeftDown);
        }
        if (Input.GetKeyUp(KeyCode.LeftControl))
        {
            Debug.Log("Up");
            MouseOperations.MouseEvent(MouseOperations.MouseEventFlags.LeftUp);
        }
    }
}
