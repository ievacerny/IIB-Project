using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class FocusControl : MonoBehaviour {

    GameObject focused_page;
    bool page_active = false;

	void Update ()
    {
        if (focused_page != null)
        {
            if (!page_active && Input.GetMouseButtonDown(0))
            {
                SetPageActivity(true);
                SetFrameTransparency(1f);
            }
            else if (page_active && Input.GetKeyDown(KeyCode.Escape))
            {
                SetPageActivity(false);
                SetFrameTransparency(0.5f);
            }
        }
	}

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Page"))
        {
            focused_page = other.gameObject;
            SetFrameTransparency(0.5f);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.gameObject.CompareTag("Page"))
        {
            SetPageActivity(false);
            SetFrameTransparency(0f);
            focused_page = null;
        }
    }

    private void SetFrameTransparency(float alpha)
    {
        GameObject frame = null;
        foreach (Transform child in focused_page.transform)
        {
            if (child.CompareTag("Frame"))
                frame = child.gameObject;
        }
        Assert.IsNotNull(frame, "No frame object associated with the page");
        Renderer frame_rend = frame.GetComponent<MeshRenderer>();

        Color new_color = frame_rend.material.color;
        new_color.a = alpha;
        frame_rend.material.color = new_color;
    }

    private void SetPageActivity(bool page_active)
    {
        this.page_active = page_active;
        PageView page_script = focused_page.GetComponent<PageView>();
        page_script.enabled = page_active;
        if (page_active)
            page_script.SetMouseCursor(transform);
    }
}
