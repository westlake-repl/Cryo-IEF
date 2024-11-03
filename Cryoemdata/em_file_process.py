
from cryosparc.dataset import Dataset

class MyEmFile(object):
    def __init__(self, emfile_path=None,selected_emfile_path=None,filetype='star'):
        self.filetype=filetype
        if emfile_path:
            if emfile_path.endswith(".star"):
                self.particles_file_content, self.particles_star_title, self.particles_id = self.read_star(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".star"):
                    self.selected_particles_file_content, self.selected_particles_star_file_title, self.selected_particles_id = self.read_star(
                        selected_emfile_path)
                    self.unselected_particles_file_content, self.unselected_particles_id = self.divide_selected_unselected_particles_star(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                else:
                    self.selected_particles_id= None

            if emfile_path.endswith(".cs"):
                self.filetype='cs'
                self.particles_file_content,  self.particles_id = self.read_cs(emfile_path)
                if selected_emfile_path is not None and selected_emfile_path.endswith(".cs"):
                    self.selected_particles_csfile_content,  self.selected_particles_id = self.read_cs(
                        selected_emfile_path)
                    self.unselected_particles_csfile_content, self.unselected_particles_id = self.divide_selected_unselected_particles_cs(
                        self.particles_file_content, self.particles_id, self.selected_particles_id)
                    # pass
                else:
                    self.selected_particles_id= None
        else:
            self.particles_id=None
            self.filetype=None
            self.selected_particles_id = None
        # pass

    def read_star(self, star_path):
        with open(star_path, "r") as starfile:
            star_data = starfile.readlines()
        for index, x in enumerate(star_data):
            if x == 'data_particles\n':
                for index2, x2 in enumerate(star_data[index:]):

                    splited_x = x2.split()
                    next_splited_x = star_data[index + index2 + 1].split()
                    if splited_x:
                        item_num = splited_x[-1].replace("#", "")
                        if item_num.isdigit():
                            if int(item_num) == len(next_splited_x) and int(item_num) != len(splited_x):
                                start_site = index + index2 + 1
                                break
        content = star_data[start_site:]
        title = star_data[:start_site]
        img_ids = self.get_star_image_id(content)
        return content, title, img_ids

    def read_cs(self,cs_path):
        cs_data=Dataset.load(cs_path)
        img_ids=cs_data['uid'].tolist()
        # mm=cs_data['blob/path'].tolist()
        # dd=cs_data['blob/idx'].tolist()
        return cs_data,img_ids
    def get_star_image_id(self, star_content):
        image_id = []
        for x in star_content:
            if len(x.strip()) > 0:
                image_id.append(x.strip().split()[5])
        return image_id

    def divide_selected_unselected_particles_star(self, particles_star_content, particles_id, selected_particles_id):
        unselected_particles_star_content = [particles_star_content[index] for index, x in enumerate(particles_id) if
                                             len(x) > 0 and x not in selected_particles_id]
        unselected_particles_id = self.get_star_image_id(unselected_particles_star_content)
        return unselected_particles_star_content, unselected_particles_id
    def divide_selected_unselected_particles_cs(self, particles_cs_content, particles_id, selected_particles_id):
        unselected_list=[]
        unselected_particles_id=[]
        for i,id in enumerate(particles_id):
            if id not in selected_particles_id:
                unselected_list.append(i)
                unselected_particles_id.append(id)
        unselected_particles_cs_content=particles_cs_content.take(unselected_list)
        return unselected_particles_cs_content, unselected_particles_id


