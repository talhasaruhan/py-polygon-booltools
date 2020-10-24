from cgeom_polygon_booltools import *
import cv2
import json

G_USER_POLYGON_SAVE_PATH = "user_polygons.json"
G_MANUAL_POLYGON_INPUT = False

g_poly_intersections = None
g_poly_done = False
g_polygons = [[]]
G_MAX_POLY_COUNT = 2
G_POLY_COLORS = [(255, 0, 0), (0, 0, 255)]

def mouse_create_polygon(event, x, y, flags, param):
    global g_polygons

    if g_poly_done:
        return

    if event == cv2.EVENT_MOUSEMOVE:
        if len(g_polygons[-1]) > 0:
            g_polygons[-1][-1] = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        if len(g_polygons[-1]) == 0:
            g_polygons[-1].append([x, y])
        g_polygons[-1].append([x, y])

def test_get_user_polygons(width = 1024, height = 1024):
    global g_poly_done, g_polygons, g_poly_intersections

    window_name = "Create Polygons"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_create_polygon)

    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    while True:
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Print lines first, so they can be painted over,
        # text and vertex markers should be at the foreground
        for poly_id, poly in enumerate(g_polygons):
            color = G_POLY_COLORS[poly_id % len(G_POLY_COLORS)]
            cv2.polylines(image, [np.array(poly, dtype=np.int32)], True, color, 1, cv2.LINE_AA)
            for x, y in poly:
                cv2.circle(image, (x, y), 4, color, 1)

        if g_poly_intersections != None:
            for x, y in intersection_points:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 0), 2)

        def print_graph_traversal_order(poly, graph, dir_db, int_vtx_txt, print_offset, prefix):
            traverse_idx = 0
            vert_idx = 0
            while True:
                if vert_idx >= intersection_vert_idx_offset:
                    is_int_pt = True
                    vert_pos = intersection_points[vert_idx-intersection_vert_idx_offset]
                else:
                    is_int_pt = False
                    vert_pos = poly[vert_idx]
                vert_pos = (int(vert_pos[0]) + print_offset[0], \
                        int(vert_pos[1]) + print_offset[1])
                if is_int_pt:
                    if vert_idx in int_vtx_txt:
                        int_vtx_txt[vert_idx].append((prefix, traverse_idx))
                    else:
                        int_vtx_txt[vert_idx] = [(prefix, traverse_idx)]
                else:
                    text = "{}-{}{}".format(vert_idx, prefix, traverse_idx)
                    cv2.putText(image, text, vert_pos,\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
                vert_idx = graph[vert_idx][1]
                traverse_idx += 1
                if vert_idx == 0:
                    break
            return int_vtx_txt

        if g_poly_done:
            if g_poly_intersections == None:
                if len(g_polygons) == 2:
                    json.dump(g_polygons, open(G_USER_POLYGON_SAVE_PATH, "w"))
                    all_data = intersection_vert_idx_offset, graph_1, graph_2,\
                            intersection_points, \
                            int_dir_db_1, int_dir_db_2, g_poly_intersections = \
                            find_polygon_intersection(g_polygons[0], g_polygons[1], is_debug=True)
                    print("Saved polygons and calculated intersections")
            else:
                # Draw intersections
                for int_idx, int_verts in enumerate(g_poly_intersections):
                    color = G_POLY_COLORS[(int_idx+5) % len(G_POLY_COLORS)]
                    cv2.fillPoly(image, pts=[int_verts], color=color)

                # Print labels for vetices that only belong to poly1 or poly2
                int_vtx_txt = {}
                print_graph_traversal_order(g_polygons[0], graph_1, int_dir_db_1, \
                        int_vtx_txt, (8, 10), "p")
                print_graph_traversal_order(g_polygons[1], graph_2, int_dir_db_2, \
                        int_vtx_txt, (8, -10), "q")
                # Print vertex labels for intersection vertices
                for vert_idx, txts in int_vtx_txt.items():
                    if len(txts) == 2 and vert_idx >= intersection_vert_idx_offset:
                        vert_pos = intersection_points[vert_idx-intersection_vert_idx_offset]
                        vert_pos = (int(vert_pos[0])+8, int(vert_pos[1])+10)
                        txts.sort()
                        prefixp, traverse_idxp = txts[0]
                        prefixq, traverse_idxq = txts[1]
                        text = "{}-{}{}-{}-{}{}-{}".format(vert_idx,\
                                prefixp, traverse_idxp,\
                                "IN" if int_dir_db_1[vert_idx] else "OUT",\
                                prefixq, traverse_idxq,\
                                "IN" if int_dir_db_2[vert_idx] else "OUT")
                        cv2.putText(image, text, vert_pos,\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

        for poly_id, poly in enumerate(g_polygons):
            color = G_POLY_COLORS[poly_id % len(G_POLY_COLORS)]
            for x, y in poly:
                cv2.circle(image, (x, y), 4, color, 1)

        if g_poly_intersections != None:
            for x, y in intersection_points:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 0), 2)

        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            g_polygons = [[]]
            g_poly_done = False
            g_poly_intersections = None

        if not g_poly_done and key == ord(" "):
            g_polygons[-1].pop()
            if len(g_polygons) < G_MAX_POLY_COUNT:
                g_polygons.append([])
            else:
                g_poly_done = True

        if key == 27:
            cv2.imwrite("vis.png", image)
            break

if G_MANUAL_POLYGON_INPUT:
    # Manual demo
    test_get_user_polygons()
else:
    # Reload last polygons
    g_polygons = json.load(open(G_USER_POLYGON_SAVE_PATH, "r"))
    g_poly_done = True
    test_get_user_polygons()
