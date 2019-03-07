using UnityEngine;

public class FocusControl : MonoBehaviour
{
    #region Private Attributes
    private GameObject focused_page;
    private LayerMask mask;
    #endregion

    #region Unity Functions

    void Start()
    {
        mask = LayerMask.GetMask("TextPlane");
    }

    void Update()
    {
        RaycastHit hit;
        // If there is a hit
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), 
                            out hit, Mathf.Infinity, mask, QueryTriggerInteraction.Collide))
        {
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * hit.distance, Color.yellow);
            // If page is the same as before, no need to do anything
            if (focused_page == hit.transform.gameObject)
            {
                return;
            }
            // If the gaze hits a new page
            else
            {
                // If there another page was activate, deactivate it first
                if (focused_page != null)
                {
                    SetPageActivity(false);
                    focused_page = null;
                }
                // Activate new page
                focused_page = hit.transform.gameObject;
                SetPageActivity(true);
            }
        }
        // If there is no hit
        else
        {
            // If a focus was on a page before, deactivate that page
            if (focused_page != null)
            {
                SetPageActivity(false);
                focused_page = null;
            }
        }
	}

    #endregion

    #region Private Methods

    private void SetPageActivity(bool page_active)
    {
        PageView page_script = focused_page.GetComponent<PageView>();
        page_script.enabled = page_active;
        if (page_active)
            page_script.SetFrameTransparency(0.5f);
        else
            page_script.SetFrameTransparency(0f);
    }

    #endregion

    #region Public Interface

    /// <summary>
    /// Returns the gameobject of the page that the gaze falls on.
    /// </summary>
    public GameObject GetActivePage()
    {
        return focused_page;
    }

    #endregion
}
