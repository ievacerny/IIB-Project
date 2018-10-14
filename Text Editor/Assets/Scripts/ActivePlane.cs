using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ActivePlane : MonoBehaviour {

    Image frame;
    InputField[] input_fields;

	// Use this for initialization
	void Start ()
    {
		frame = this.gameObject.GetComponent<Image>();
        input_fields = this.gameObject.transform.GetComponentsInChildren<InputField>();

        //// Add listeners that would automatically enable and disable the frame
        //foreach (InputField field in input_fields)
        //{
        //    field.onEndEdit.AddListener(delegate { frame.enabled = false; });
        //}
    }
	
	// Update is called once per frame
	void Update ()
    {
        DisableFrame();
        EnableFrame();
	}

    void EnableFrame ()
    {
        if (frame.enabled)
            return;

        foreach (InputField field in input_fields)
        {
            if (field.isFocused)
            {
                frame.enabled = true;
                Debug.Log("Enabled");
                break;
            }
        }
    }

    void DisableFrame ()
    {
        if (!frame.enabled)
            return;

        bool all_disabled = true;
        foreach (InputField field in input_fields)
            all_disabled &= !field.isFocused;
        
        if (all_disabled)
        {
            frame.enabled = false;
            Debug.Log("Disabled");
        }
        
    }

}
