import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.putText(frame, 'STARTING!!!!!!!!!', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)            
# Show to screen
cv2.imshow('OpenCV Feed', frame)
cv2.waitKey(500)
# Loop through sequences aka videos
for sequence in range(30):
    # Loop through video length aka sequence length
    for frame_num in range(60):
        # Read feed
        ret, frame = cap.read()
        
        # NEW Apply wait logic
        if frame_num == 0: 
            cv2.putText(frame, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, 'Video Number {}'.format(sequence), (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('OpenCV Feed', frame)
        else: 
            cv2.putText(frame, 'Video Number {}'.format(sequence), (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('OpenCV Feed', frame)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
                    
cap.release()
cv2.destroyAllWindows()