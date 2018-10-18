using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class ActivePage : MonoBehaviour {

    public bool inEditMode;

    public GameObject frame_quad;
    private Renderer frame_quad_rend;
    private Transform cursor_transform;

    private bool cursor_on_page = false;

    private float margin = 0.00575f;
    private float width = 0.0186f;
    private float height = 0.0643f;

    ModelViewMapping modelview;

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

        modelview = new ModelViewMapping(margin, width, height);
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

        if (cursor_transform != null)
        {
            Vector3 coords = transform.InverseTransformPoint(cursor_transform.position);
            if (coords.x <= 0.5f && coords.x >= -0.5f && coords.y <= 0.5 && coords.y >= -0.5f)
            {
                var ind = modelview.CoordsToIndex(coords);
                var coords2 = modelview.IndexToCoords(ind);
                Debug.Log(string.Format("Real {0}, new {1}; row {2}, col {3}", coords.ToString("F4"), coords2.ToString("F4"), ind.row, ind.col));
            }
        }
        

    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.name == "FingerPoint")
        {
            cursor_on_page = true;
            cursor_transform = other.transform;

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
            cursor_transform = null;

            if (!inEditMode)
            {
                Color new_color = frame_quad_rend.material.color;
                new_color.a = 0f;
                frame_quad_rend.material.color = new_color;
            }
        }
    }

}
