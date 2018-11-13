using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class FocusControl : MonoBehaviour {

    GameObject focused_page;

	void Update ()
    {
        RaycastHit hit;
        if (Physics.Raycast(transform.position, 
                            transform.TransformDirection(Vector3.forward), 
                            out hit, Mathf.Infinity, ~0, QueryTriggerInteraction.Collide))
        {
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * hit.distance, Color.yellow);
            if (focused_page == hit.transform.gameObject)
            {
                return;
            }
            else
            {
                Debug.Log(hit.transform.gameObject.name);
                if (focused_page != null)
                {
                    SetPageActivity(false);
                    focused_page = null;
                }
                focused_page = hit.transform.gameObject;
                SetPageActivity(true);
            }
        }
        else
        {
            if (focused_page != null)
            {
                SetPageActivity(false);
                focused_page = null;
            }
        }
	}

    private void SetPageActivity(bool page_active)
    {
        PageView page_script = focused_page.GetComponent<PageView>();
        page_script.enabled = page_active;
        if (page_active)
        {
            //page_script.SetMouseCursor(transform);
            page_script.SetFrameTransparency(0.5f);
        }
        else
        {
            page_script.SetFrameTransparency(0f);
        }
    }
}
