using UnityEngine;

public class FingerPointFollowCursor : MonoBehaviour
{
    private Transform cursor_object_transf;
    private float z_pos;

    void Start()
    {
        cursor_object_transf = GetComponent<Transform>();
        z_pos = cursor_object_transf.position.z - Camera.main.transform.position.z;
        Cursor.visible = false;
	}
	
	void Update()
    {
        cursor_object_transf.position = Camera.main.ScreenToWorldPoint(
            new Vector3(Input.mousePosition.x, Input.mousePosition.y, z_pos));
    }

    void OnApplicationQuit()
    {
        Cursor.visible = true;
    }
}
