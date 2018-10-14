using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class ActivePage : MonoBehaviour {

    public bool inEditMode;

    public GameObject frame_quad;
    private Renderer frame_quad_rend;

    private bool cursor_on_page = false;

    // Use this for initialization
    void Start ()
    {
        if (!frame_quad)
        {
            // There should be only one frame in a page. If not, it will grab the first one
            foreach (Transform child in transform) if (child.CompareTag("Frame"))
                    frame_quad = child.gameObject;
        }

        Assert.IsNotNull(frame_quad);
        frame_quad_rend = frame_quad.GetComponent<MeshRenderer>();
        Color new_color = frame_quad_rend.material.color;
        new_color.a = 0f;
        frame_quad_rend.material.color = new_color;
    }
	
	// Update is called once per frame
	void Update ()
    {
        if (!inEditMode && cursor_on_page && Input.GetMouseButtonDown(0))
        {
            inEditMode = true;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 1f;
            frame_quad_rend.material.color = new_color;
        }
        else if (inEditMode && cursor_on_page && Input.GetKeyDown(KeyCode.Escape))
        {
            inEditMode = false;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 0.5f;
            frame_quad_rend.material.color = new_color;
        }
        else if (inEditMode && !cursor_on_page && Input.GetKeyDown(KeyCode.Escape))
        {
            inEditMode = false;
            Color new_color = frame_quad_rend.material.color;
            new_color.a = 0f;
            frame_quad_rend.material.color = new_color;
        }
		
	}

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.name == "FingerPoint")
        {
            cursor_on_page = true;

            if (!inEditMode)
            {
                Color new_color = frame_quad_rend.material.color;
                new_color.a = 0.5f;
                frame_quad_rend.material.color = new_color;
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.name == "FingerPoint")
        {
            cursor_on_page = false;

            if (!inEditMode)
            {
                Color new_color = frame_quad_rend.material.color;
                new_color.a = 0f;
                frame_quad_rend.material.color = new_color;
            }
        }
    }

}
